import numpy as np
import time

def _parse_rigol_preamble(preamble: str):
    # Typical Rigol preamble is comma-separated numeric fields.
    # Common DS series format (varies by model/firmware):
    # FORMAT,TYPE,POINTS,COUNT,XINCR,XORIG,XREF,YINCR,YORIG,YREF
    parts = preamble.strip().split(',')
    if len(parts) < 10:
        raise ValueError(f"Unexpected preamble: {preamble!r}")

    xincr = float(parts[4])
    xorig = float(parts[5])
    xref  = float(parts[6])
    yincr = float(parts[7])
    yorig = float(parts[8])
    yref  = float(parts[9])
    points = int(float(parts[2]))
    return points, xincr, xorig, xref, yincr, yorig, yref

def read_waveform(scope, trigger_channel = 1, read_channel = [1,2], acquire_length = 4096, 
                  calibrate = True, initialize = False,
                  calibration_data=None, trigger_mode="normal", wait_time_per_channel = 0.4, 
                  trigger_timeout=10):

    # 2. Start Acquisition
    scope.write(":SINGle")
    scope.write(":SINGle")
    
    # 3. Wait for the Scope to START acquisition
    start_time = time.time()
    while True:
        state = scope.query(":TRIGger:STATus?").strip() 
        if state in ["TD", "RUN", "WAIT", "AUTO"]:
            break
        if (time.time() - start_time) > trigger_timeout:
            print(f"[ERROR] Timeout: Scope stayed in {state} state.")
            return {}, []

    # 4. Wait for the Scope to STOP acquisition
    while True:
        state = scope.query(":TRIGger:STATus?").strip()
        if state == "STOP":
            break
        if (time.time() - start_time) > trigger_timeout:
            print(f"[ERROR] Timeout: Scope stayed in {state} state.")
            return {}, []

    # 5. Wait some time for the memory to be ready.
    time.sleep(wait_time_per_channel*len(read_channel))
        
            
    data={}
    for ch in read_channel:
        scope.write(f":WAVeform:SOURce CHANnel{ch}")# Select source.    
        scope.write(":WAVeform:MODE RAW") # RAW: enable reading all points from memory, up to memory depth
        scope.write(":WAVeform:FORMat WORD") 
        scope.write(f":WAVeform:POINts {acquire_length}")
        # Copy the waveform

        # Request + read data as one transaction (this is the key fix)
        varWavData = np.array(scope.query_binary_values(":WAV:DATA?", datatype='H',delay =0.15,
                                        container=np.array, is_big_endian=False), dtype=float)
            
        
        if calibrate:
            # Calibration
            if calibration_data is None:
                # Fetch scaling once per channel (some scopes tie Y scaling to source)
                pre = scope.query(":WAV:PRE?")
                points_reported, dx, x0, xref, dy, y0, yref = _parse_rigol_preamble(pre)
            else:
                dx,dy,x0,y0 = calibration_data[ch]
            varWavData = (varWavData-2**15 + y0)*dy 
            time_series = np.arange(len(varWavData))*dx + x0
        else:
            time_series = np.arange(len(raw))
        # Float32 is enough for 16-bit integer    
        data[ch] = varWavData.astype(np.float32)
    scope.write(":RUN");
        
    return data, time_series   