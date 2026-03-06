import math, sys, time
import struct
from importlib import reload
from tqdm import tqdm

import pyvisa
import numpy as np

# import helper_visa as vs
# import helper_basic as hp


class siglent_sds:
    tdiv_enum = [200e-12, 500e-12, 1e-9,
                 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
                 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
                 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
                 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    def __init__(self, my_instrument, address=None, timeout=5000, idn=False):
        # self.rm = pyvisa.ResourceManager()
        # self.rm.list_resources()
        # self.my_instrument = self.rm.open_resource(address)
        # self.my_instrument = connect(address=address, timeout=timeout, idn=idn)
        self.my_instrument = my_instrument
        print(self.my_instrument.query('*IDN?'))
        return

    @staticmethod
    def main_desc(recv):
        param_addr_type = {"data_bytes": [0x3c, "i"],
                           "point_num": [0x74, 'i'],
                           "fp": [0x84, 'i'],
                           "sp": [0x88, 'i'],
                           "vdiv": [0x9c, 'f'],
                           "offset": [0xa0, 'f'],
                           "code": [0xa4, 'f'],
                           "adc_bit": [0xac, 'h'],
                           "interval": [0xb0, 'f'],
                           "delay": [0xb4, 'd'],
                           "tdiv": [0x144, 'h'],
                           "probe": [0x148, 'f']}
        data_byte = {"i": 4, "f": 4, "h": 2, "d": 8}
        param_val = {}
        for key, addr_type in param_addr_type.items():
            addr_start = addr_type[0]
            _format = addr_type[1]
            _bytes = recv[addr_start:addr_start+data_byte[_format]]
            param_val[key] = struct.unpack(_format, _bytes)[0]
        if "tdiv" in param_val:
            param_val["tdiv"] = siglent_sds.tdiv_enum[param_val["tdiv"]]
        if "vdiv" in param_val:
            param_val["vdiv"] = param_val["vdiv"]*param_val["probe"]
        if "offset" in param_val:
            param_val["offset"] = param_val["offset"]*param_val["probe"]
        return param_val

    def get_fft(self, FUNC=1, plot=True, log=False):
        """
        Read FFT data from Siglent scope

        Return:

        """
        # Get the channel waveform parameter data blocks and parse them
        self.my_instrument.write(f"WAV:SOUR F{FUNC}")
        self.my_instrument.write("WAV:PREamble?")
        recv_all = self.my_instrument.read_raw()
        recv = recv_all[recv_all.find(b'#') + 11:]
        param_val = self.main_desc(recv)
        display_len = int(param_val["delay"] / param_val["interval"])+1
        unit = self.my_instrument.query(
            f"FUNC{FUNC}:FFT:UNIT?").strip()  # {Vrms,DBm,DBVrms}
        if unit == "DBm":
            load = float(self.my_instrument.query(
                f"FUNC{FUNC}:FFT:LOAD?").strip())
        # {NORMal|MAXHold|AVERage[,num]}
        mode = self.my_instrument.query(f"FUNC{FUNC}:FFT:MODE?").strip()
        # Get the waveform data
        self.my_instrument.write("WAV:DATA?")
        recv_all = self.my_instrument.read_raw().rstrip()
        block_start = recv_all.find(b'#')
        data_digit = int(recv_all[block_start + 1:block_start + 2])
        data_start = block_start + 2 + data_digit
        recv = recv_all[data_start:]
        # Unpack data.
        volt_value = []
        freq_value = []
        len_data = int(len(recv) / 8)

        print("FFT length:", len_data)
        data_float = np.frombuffer(recv, dtype=np.float32)
        data_real = data_float[::2]
        data_imag = data_float[1::2]
        data_freq = np.arange(1, len_data+1)*param_val["interval"]
        if mode == "NORMal":
            data_plot = np.sqrt(data_real**2 + data_imag ** 2)
        else:
            data_plot = data_real

        # if unit == "DBVrms":
        #     data_float = 20*math.log10(data_float)
        # elif unit == "DBm":
        #     data_float = 10 * math.log10(data_float*data_float/load/1E-3)

        # print(recv[0:4])
        # for i in range(0, len_data):
        #     data_real = struct.unpack("f", recv[8 * i:8 * i + 4])
        #     data_imag = struct.unpack("f", recv[8 * i + 4:8 * i + 8])
        #     data_real = list(data_real)[0]
        #     data_imag = list(data_imag)[0]
        #     if mode == "NORMal":
        #         data_float = math.sqrt(pow(float(data_real), 2) + pow(float(data_imag), 2))
        #     else:
        #         data_float = float(data_real)
        #     if unit == "DBVrms":
        #         data_float = 20*math.log10(data_float)
        #     elif unit == "DBm":
        #         data_float = 10 * math.log10(data_float*data_float/load/1E-3)
        #     volt_value.append(data_float)
        #     freq_value.append(i*param_val["interval"])

        # Draw Waveform
        if plot:
            plt.figure(figsize=(7, 5))
            plt.plot(data_freq, data_plot, markersize=2)
            plt.grid()
            plt.xlabel("Freq [Hz]")
            plt.ylabel(f"[V]")
            plt.show()

        return data_freq, data_real, data_imag
    
    
    def get_trace_screen(self, CHANNEL="C1"):
        sds = self.my_instrument
        sds.timeout = 2000  # default value is 2000(2s)
        # default value is 20*1024(20k bytes)
        sds.chunk_size = 20 * 1024 * 1024
        # Get the channel waveform parameter data blocks and parse them
        sds.write(":WAVeform:STARt 0")
        sds.write("WAV:SOUR {}".format(CHANNEL))
        sds.write("WAV:PREamble?")
        recv_all = sds.read_raw()

        recv = recv_all[recv_all.find(b'#') + 11:]
        param_dic = self.main_desc(recv)
        # Get the waveform points and confirm the number of waveform slice reads
        points = param_dic["point_num"]
        one_piece_num = float(sds.query(":WAVeform:MAXPoint?").strip())
        read_times = math.ceil(points / one_piece_num)
        # Set the number of read points per slice, if the waveform points is greater than the maximum
        # number of slice reads
        if points > one_piece_num:
            sds.write(":WAVeform:POINt {}".format(one_piece_num))
        # Choose the format of the data returned
        sds.write(":WAVeform:WIDTh BYTE")
        if param_dic["adc_bit"] > 8:
            sds.write(":WAVeform:WIDTh WORD")
        # Get the waveform data for each slice
        recv_byte = b''
        for i in range(0, read_times):
            start = i * one_piece_num
            # Set the starting point of each slice
            sds.write(":WAVeform:STARt {}".format(start))
            # Get the waveform data of each slice
            sds.write("WAV:DATA?")
            recv_rtn = sds.read_raw()
            # Splice each waveform data based on data block information
            block_start = recv_rtn.find(b'#')
            data_digit = int(recv_rtn[block_start + 1:block_start + 2])
            data_start = block_start + 2 + data_digit
            data_len = int(recv_rtn[block_start + 2:data_start])
            recv_byte += recv_rtn[data_start:data_start + data_len]
            
        # Unpack signed byte data.
        if param_dic["adc_bit"] > 8:
            convert_data = struct.unpack("%dh" % points, recv_byte)
        else:
            convert_data = struct.unpack("%db" % points, recv_byte)
        del recv_byte
        gc.collect()
        
        # Calculate the voltage value and time value
        
        # Calculate the voltage value and time value
        volt_value = np.array(convert_data)/param_dic["code"]*param_dic["vdiv"]  -param_dic["offset"]
        time_value =  - (param_dic["tdiv"] * 10 / 2) + np.arange(len(convert_data)) * param_dic["interval"] + param_dic["delay"]
        
        return time_value, volt_value



class Source:
    def __init__(self, address="USB0::0xF4EC::0x1305::SSA3PCEX6R0308::INSTR"):
        # print(address)
        rm = pyvisa.ResourceManager()
        self.resource_list = rm.list_resources()
        self.UPDATED = False

        self.config = {\
            "General":{
                "Info":
                """Name: Siglent SSA3000X
Support mode: sweeping
Frequency range: 9 kHz-3.5 GHz
""",
                "Supported Mode":["SWEEP"],
                "Frequency Limit":[9_000, 3_500_000_000],     
            },
            "Frequency":{
                "Start": 70_000_000,
                "Stop": 130_000_000,
                "Center": 100_000_000,
                "Span": 60_000_000,    
            },
            "Bandwidth":{
                ""

            },
            "Amplitude":{

            },
            "Detect":{

            },
            "Display":{

            },                                                
        }

    def connect(self, resource_idx):
        try:
            self.scope = vs.connect(address = self.resource_list[resource_idx], timeout=10_000, idn=False) # set 10 second timeout  
            self.scope.write(":FORMat REAL,32")
        except:
            self.scope = None

    def getTrace(self):  
        trace =  vs.ssa3000_gettrace(self.scope)
        freq_range = vs.ssa3000_getfreq(self.scope)
        freq = np.linspace(freq_range[0], freq_range[1], len(trace), endpoint=True)

        self.freq = freq
        self.trace = trace
        self.UPDATED = True

        return freq, trace
    
    def getFreq(self):
        self.freq = vs.ssa3000_gettrace(self.scope)
        return self.freq
    



class vs:
    """
    Helper class for visa related communications
    """
    @staticmethod
    def connect(address=None, timeout = 5000, idn=False):
        rm = pyvisa.ResourceManager()
        
        # List all resources if address is not given
        if address is None:
            resource_list = rm.list_resources()

            # Try to get the *IDN from each equipment
            for i in range(len(resource_list)):
                # print(my_instrument.query('*IDN?'))
                try:
                    if idn:
                        my_instrument = rm.open_resource(resource_list[i],open_timeout=500)  
                        my_instrument_name = my_instrument.query('*IDN?')
                        # disconnect(my_instrument)
                        my_instrument.close()
                    else:
                        my_instrument_name=""
                    print(f"[{i}]: {resource_list[i]} ", my_instrument_name)
                except:
                    print(f"[{i}]: {resource_list[i]} cannot be opened")

            # Select one and open    
            i_instrument = input(prompt="Type number to connect:")
            address = resource_list[int(i_instrument)]
            
        my_instrument = rm.open_resource(address, open_timeout=4000)  
        my_instrument_name = my_instrument.query('*IDN?')
        # my_instrument.write("*RST"); # Reset the function generator
        # my_instrument.clear();   # Clear the buffer
        my_instrument.timeout = timeout # Set  timeout
        # my_instrument.write(":DISP ON")
            
        print(f"Connected to VISA [{address}]: ", my_instrument_name)    

        return my_instrument

    @staticmethod
    def ssa3000_gettrace(scope, n_queries = 100):
        """
        scope.write(":FORMat ASCII")
        a=scope.query(":TRACe? 1")
        """
        # try:
        #     scope.continuous == "OFF"
        # except:
        # scope.write(":INITiate:CONTinuous OFF")
        # scope.continuous = "OFF"
            
        scope.write(":INITiate")
        
        
        while n_queries>0:
            if scope.query("*OPC?")=='1\n':
                break    
                
        scope.write(":TRACe? 1")
        trace_raw = scope.read_raw()[:-1] # delete the last \n char
        return np.frombuffer(trace_raw, dtype=np.float32)      


    @staticmethod
    def ssa3000_getfreq(scope):
        f_start = float(scope.query(":FREQuency:STARt?")[:-1])
        f_stop = float(scope.query(":FREQuency:STOP?")[:-1])
        return f_start,f_stop          
