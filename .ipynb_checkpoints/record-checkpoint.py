

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple, Union, Dict, Any

import numpy as np
import h5py


TraceDict = Dict[str, np.ndarray]
GetTraceReturn = Union[
    TraceDict,
    Tuple[TraceDict, float],
    Tuple[TraceDict, float, Dict[str, Any]],
]


def record_traces_dict_to_hdf5(
    filename: str,
    get_trace: Callable[[], GetTraceReturn],
    *,
    max_traces: Optional[int] = None,
    duration_s: Optional[float] = None,
    flush_every_traces: int = 10,
    flush_every_s: float = 5.0,
    group_path: str = "/waveforms",
    timestamps_path: str = "/timestamps",
    dtypes: Optional[Dict[str, np.dtype]] = None,
    chunk_traces: int = 1,
    points_per_trace: Optional[int] = None,
    store_metadata: bool = True,
    file_attributes: Optional[Dict[str, Any]] = None,
    require_same_channels: bool = True,
) -> None:
    """
    Records oscilloscope traces to an HDF5 file incrementally (no compression),
    where get_trace() returns a dict: {channel_name: 1D array}.

    Satisfies:
      1) Creates an HDF5 file (overwrites), no compression.
      2) Pulls traces via get_trace() and writes incrementally per channel.
      3) Periodically flush() by trace count and/or elapsed time.

    get_trace() may return:
      - trace_dict
      - (trace_dict, timestamp)
      - (trace_dict, timestamp, meta_dict)

    File layout:
      /waveforms/<channel_name>   dataset shape (N_records, N_points)
      /timestamps                dataset shape (N_records,)
      /trace_meta/<i>            optional group of per-trace attrs

    Assumptions (default):
      - All channels present in first trace persist for the run (require_same_channels=True).
      - All channels have same number of points per trace (common on scopes).
    """
    if max_traces is None and duration_s is None:
        # runs forever until interrupted
        pass
    if flush_every_traces < 1:
        raise ValueError("flush_every_traces must be >= 1")
    if flush_every_s <= 0:
        raise ValueError("flush_every_s must be > 0")
    if chunk_traces < 1:
        raise ValueError("chunk_traces must be >= 1")

    start_time = time.time()
    last_flush_time = start_time

    def _unpack(result: GetTraceReturn) -> Tuple[TraceDict, float, Optional[Dict[str, Any]]]:
        if isinstance(result, tuple):
            if len(result) == 2:
                trace_dict, ts = result
                meta = None
            elif len(result) == 3:
                trace_dict, ts, meta = result
            else:
                raise ValueError("get_trace() tuple must be (trace_dict,), (trace_dict, ts), or (trace_dict, ts, meta)")
        else:
            trace_dict = result
            ts = time.time()
            meta = None

        if not isinstance(trace_dict, dict) or not trace_dict:
            raise ValueError("get_trace() must return a non-empty dict: {channel_name: 1D array}")

        # Ensure arrays are numpy arrays and 1D
        out: TraceDict = {}
        for ch, arr in trace_dict.items():
            # if not isinstance(ch, str) or not ch:
            #     raise ValueError(f"Channel name must be a non-empty str. Got: {ch!r}")
            a = np.asarray(arr)
            if a.ndim != 1:
                raise ValueError(f"Channel {ch} trace must be 1D. Got shape {a.shape}")
            out[f"{ch}"] = a
        return out, float(ts), meta

    with h5py.File(filename, "w") as f:
        print("File created:", filename)
        # Optional run-level attributes
        if store_metadata and file_attributes:
            for k, v in file_attributes.items():
                f.attrs[k] = v

        # Take first acquisition to infer channel list, point count, and dtypes
        tmp_data = get_trace()
        first_td, first_ts, first_meta = _unpack(tmp_data)

        channels = list(first_td.keys())
        if require_same_channels:
            channels_set = set(channels)

        # Infer points per trace
        first_lengths = {ch: int(first_td[ch].size) for ch in channels}
        if points_per_trace is None:
            # Require all channels same length
            lengths_set = set(first_lengths.values())
            if len(lengths_set) != 1:
                raise ValueError(f"Channels have different lengths in first trace: {first_lengths}")
            n_points = lengths_set.pop()
        else:
            n_points = int(points_per_trace)
            for ch, n in first_lengths.items():
                if n != n_points:
                    raise ValueError(f"points_per_trace={n_points} does not match first trace length for {ch}={n}")

        # Create group and per-channel datasets
        wf_grp = f.create_group(group_path)

        # Decide dtype per channel
        dtypes = dtypes or {}
        ds_by_ch: Dict[str, h5py.Dataset] = {}

        for ch in channels:
            ds_dtype = np.dtype(dtypes[ch]) if ch in dtypes else first_td[ch].dtype
            ds = wf_grp.create_dataset(
                ch,
                shape=(0, n_points),
                maxshape=(None, n_points),
                dtype=ds_dtype,
                chunks=(chunk_traces, n_points),
                compression=None,  # no compression
                shuffle=False,
                fletcher32=False,
            )
            ds_by_ch[ch] = ds

        # Timestamps dataset
        ts_ds = f.create_dataset(
            timestamps_path,
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(chunk_traces, 1024),),
            compression=None,
        )

        # Optional per-trace meta
        meta_grp = f.create_group("/trace_meta") if store_metadata else None

        n_written = 0

        def _append(trace_dict: TraceDict, ts: float) -> None:
            nonlocal n_written

            # Optional strict channel checking
            if require_same_channels:
                if set(trace_dict.keys()) != channels_set:
                    missing = channels_set - set(trace_dict.keys())
                    extra = set(trace_dict.keys()) - channels_set
                    raise ValueError(f"Channel set changed. Missing={sorted(missing)} Extra={sorted(extra)}")
            else:
                # If channels change and you're OK with it, you could create new datasets here.
                # For simplicity, we enforce that all expected channels exist.
                for ch in channels:
                    if ch not in trace_dict:
                        raise ValueError(f"Missing expected channel {ch} in trace_dict")

            # Extend timestamps
            ts_ds.resize((n_written + 1,))
            ts_ds[n_written] = ts

            # Extend and write each channel
            for ch in channels:
                arr = trace_dict[ch]
                if arr.size != n_points:
                    raise ValueError(f"Trace length changed for {ch}: expected {n_points}, got {arr.size}")

                ds = ds_by_ch[ch]
                ds.resize((n_written + 1, n_points))

                # Cast if needed
                if arr.dtype != ds.dtype:
                    arr = arr.astype(ds.dtype, copy=False)

                ds[n_written, :] = arr

            n_written += 1

        # Write first record
        _append(first_td, first_ts)

        # Store run-level metadata from first_meta if supplied
        if store_metadata and isinstance(first_meta, dict) and first_meta:
            for k, v in first_meta.items():
                if k not in f.attrs:
                    try:
                        f.attrs[k] = v
                    except TypeError:
                        f.attrs[k] = str(v)

        # Flush after initial write
        f.flush()
        last_flush_time = time.time()

        try:
            while True:
                print(f"N traces: {n_written}, duration {time.time() - start_time:.1f}s", end="\r")
                if max_traces is not None and n_written >= max_traces:
                    break
                if duration_s is not None and (time.time() - start_time) >= duration_s:
                    break

                for i in range(4):
                    # tmp_data = get_trace()
                    td, ts, meta = _unpack(get_trace())
                    if len(list(td.keys()))>0:
                        break
                _append(td, ts)

                # Optional per-trace metadata
                if store_metadata and isinstance(meta, dict) and meta:
                    g = meta_grp.create_group(str(n_written - 1))
                    for k, v in meta.items():
                        try:
                            g.attrs[k] = v
                        except TypeError:
                            g.attrs[k] = str(v)

                # Periodic flush (count or time)
                now = time.time()
                if (n_written % flush_every_traces) == 0 or (now - last_flush_time) >= flush_every_s:
                    f.flush()
                    last_flush_time = now

        except KeyboardInterrupt:
            pass
        finally:
            f.flush()

    print("Finished")