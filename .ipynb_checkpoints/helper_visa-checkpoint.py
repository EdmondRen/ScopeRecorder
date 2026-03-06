# import pandas as pd
import pyvisa as visa
import pyvisa
import string
import struct
import sys
import time,datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
from tqdm import tqdm
import os
import types
import platform
from pylab import *
from helper_basic import *
import helper_basic as hp
debug=  False


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
                    disconnect(my_instrument)
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
    
    
    #--------------------------------------------\
    # Add some helper functions/ I don't know when they could be useful
    # =========================================================
    # Check for instrument errors:
    def check_instrument_errors(self, command):
        while True:
            error_string = self.query(":SYSTem:ERRor?")
            if error_string:   # If there is an error string value.
                if error_string.find("+0,", 0, 3) == -1:   # Not "No error".
                    print("ERROR: %s, command: '%s'" % (error_string, command))
                    print("Exited because of error.")
                    sys.exit(1)
                else:   # "No error"
                    break
            else:   # :SYSTem:ERRor? should always return string.
                print("ERROR: :SYSTem:ERRor? returned nothing, command: '%s'" % command)
                print("Exited because of error.")
                sys.exit(1)   
    my_instrument.check_instrument_errors = types.MethodType(check_instrument_errors,my_instrument)
                
    
    # =========================================================
    # Send a command and check for errors:
    def do_command(self, command, hide_params=False):
        if hide_params:
            (header, data) = command.split(" ", 1)
        if debug:
            print("\nCmd = '%s'" % header)
        else:
            if debug:
                print("\nCmd = '%s'" % command)
        self.write("%s" % command)
        if hide_params:
            self.check_instrument_errors(header)
        else:
            self.check_instrument_errors(command)
    my_instrument.do_command = types.MethodType(do_command,my_instrument)
            

    # =========================================================
    # Send a command and binary values and check for errors:
    def do_command_ieee_block(self, command, values):
        if debug:
            print("Cmb = '%s'" % command)
        self.write_binary_values("%s " % command, values, datatype='B')
        self.check_instrument_errors(command)
    my_instrument.do_command_ieee_block = types.MethodType(do_command_ieee_block,my_instrument)

    # =========================================================
    # Send a query, check for errors, return string:
    def do_query_string(self, query):
        if debug:
            print("Qys = '%s'" % query)
        result = self.query("%s" % query)
        self.check_instrument_errors(query)
        return result
    my_instrument.do_query_string = types.MethodType(do_query_string,my_instrument)

    # =========================================================
    # Send a query, check for errors, return floating-point value:
    def do_query_number(self, query):
        if debug:
            print("Qyn = '%s'" % query)
        results = self.query("%s" % query)
        self.check_instrument_errors(query)
        return float(results)
    my_instrument.do_query_number = types.MethodType(do_query_number,my_instrument)

    # =========================================================
    # Send a query, check for errors, return binary values:
    def do_query_ieee_block(self, query):
        if debug:
            print("Qys = '%s'" % query)
        result = self.query_binary_values("%s" % query, datatype='s')
        self.check_instrument_errors(query)
        return result[0]
    my_instrument.do_query_ieee_block = types.MethodType(do_query_ieee_block,my_instrument)    
    
    return my_instrument


def disconnect(my_instrument):
    my_instrument.write(":SYSTem:COMMunicate:RLSTate LOCal")
    
    
    
def get_calibration(scope, read_channel = [1,2]):
    calibration_data = {}
    
    for ch in read_channel:
        # Copy the waveform
        scope.write(f":WAVeform:SOURce CHANnel{ch}")# Select source.
        
        dx = float(scope.query(":WAVeform:XINCrement?")[:-1])
        dy = float(scope.query(":WAVeform:YINCrement?")[:-1])
        x0 = float(scope.query(":WAVeform:XORigin?")[:-1])
        y0 = float(scope.query(":WAVeform:YORigin?")[:-1])
        calibration_data[ch] = [dx,dy,x0,y0]
    
    return calibration_data

def read_waveform(scope, trigger_channel = 1, read_channel = [1,2], acquire_length = 4096, calibrate = True, initialize = False, calibration_data=None, trigger_mode="normal"):

    if initialize:
        # Trigger
        scope.write(":TRIGger:MODE EDGE")
        scope.write(f":TRIGger:EDGE:SOURce CHANnel{trigger_channel}") #CHANNEL 1 TRIGGER
        # scope.write(":TRIGger:EDGE:SLOPe NEGative")

        # Acquisation setup
        if trigger_mode == "normal":
            scope.write(":ACQuire:TYPE NORMal")
        else:
            scope.write(":ACQuire:TYPE AUTO")

        scope.write(":ACQuire:MODE RTIMe") # Realtime acquire mode
        scope.write(":ACQuire:INTerpolate 0") # Disable interpolation
        scope.write(f":ACQuire:POINts:ANALog {acquire_length}")
        
        # Other settings
        scope.write(":SYSTem:HEADer OFF")# ' Response headers off.
        scope.write(":WAVeform:FORMat WORD")# ' Select word format.
        scope.write(":WAVeform:BYTeorder LSBFirst")# ' Select word format.
        scope.write(":WAVeform:STReaming 0")#        
        
        for ch in read_channel:
            scope.write(f":CHANnel{ch}:DISPlay ON")        
        
        

    # Digitize, and display
    # The command runs x10 faster without parameters... Keep the digitize command as is and set the channel on scope
    digi_command = ":DIGitize "
    # for ch in read_channel:
    #     digi_command+=f"CHANnel{ch},"
    # digi_command = digi_command[:-1]
    scope.write(digi_command);

    data={}
    for ch in read_channel:

        # Copy the waveform
        scope.write(f":WAVeform:SOURce CHANnel{ch}")# Select source.
        scope.write(":WAVeform:DATA?")#
        varWavData = np.array(scope.read_binary_values(datatype='h'))
        
        if calibrate:
            # Calibration
            if calibration_data is None:
                dx = float(scope.query(":WAVeform:XINCrement?")[:-1])
                dy = float(scope.query(":WAVeform:YINCrement?")[:-1])
                x0 = float(scope.query(":WAVeform:XORigin?")[:-1])
                y0 = float(scope.query(":WAVeform:YORigin?")[:-1])
            else:
                dx,dy,x0,y0 = calibration_data[ch]
            varWavData = varWavData*dy + y0
            time_series = np.arange(len(varWavData))*dx + x0
        else:
            time_series = np.arange(len(varWavData))
        # Float32 is enough for 16-bit integer    
        data[ch] = varWavData.astype(np.float32)
        
    return data, time_series    


def get_events(scope, Nevents = 100, trigger_channel = 1, read_channel = [1,2], acquire_length = 4096, calibrate = True, print_every_n = 10):
    # Test:
    # data_save = get_events(scope, Nevents = 10, trigger_channel = 1, read_channel = [1,2], acquire_length = 4096, calibrate = True, print_every_n = 1)

    n_acquired = 0 
    data_save = {}
    for ch in read_channel:
        data_save[ch]=[]
        
        
    # Initialize settings:
    data,time_series=read_waveform(scope, trigger_channel = trigger_channel, read_channel = read_channel, acquire_length = acquire_length, calibrate = False, initialize = True, calibration_data=None)
    # Get the calibration 
    calibration_data = get_calibration(scope)
    
    
    # Start repetitive acquisation
    t1=time.time()   
    t2a=t1
    while(n_acquired<Nevents):
        try:
            data, time_series=read_waveform(scope, trigger_channel = trigger_channel, read_channel = read_channel, acquire_length = acquire_length, calibrate = calibrate, initialize = False, calibration_data=calibration_data)
            for i, ch in enumerate(read_channel):
                data_save[ch].append(data[ch][:])
            n_acquired+=1  
            if n_acquired%print_every_n ==0:
                t2b=time.time()
                print(f"{n_acquired}/{Nevents} events acquired, time elapsed {t2b - t1:.1f} s, time from last print {t2b-t2a:.1f} s")
                t2a=t2b
                
            del data
        except KeyboardInterrupt:
            print("  KeyboardInterrupt. You pressed ctrl c...")
            break                
        except Exception as e: # Any other exception
            print("  Exception:", str(e)) # Displays the exception without raising it
            continue
            
    data_save["metadata"]={}
    data_save["metadata"]["time_series"]=time_series
    return data_save
    

    
    
def upload_waveform(funcgen, waveform_int, waveform_duration = 0.4, ch = 1, interpolation=True, trigger_mode = "INTernal2", trigger_freq = 500, output_voltage = 0.1, output_offset=0, output_impedance=50, RESET=True, INIT = True):
    """
    Upload waveform to Keysight 81160A function generator
    
    
    funcgen: visa object
        
    waveform_int: list of int
        list of waveform in integer within range, -8192 to 8192
        The length of this waveform is limited to:
            - 16384:
            - 262144: single channel
    waveform_duration: float
        time of the entire waveform [nano second]
    ch: int
        1 or 2
    interpolation: bool
        Turn on interpolation or not
    trigger_mode: str
        one of "IMMediate|INTernal1|INTernal2|EXTernal|BUS|MANual"
        IMMediate: continuous
        INTernal2: triggered
    trigger_freq: int
        Internal trigger frequency
        
    output_voltage: float
        output voltage in VPP
    output_offset: float
        output voltage offset, default is 0
    output_impedance: int
        output impedance in Ohm. You can set the load to any value from 0.3 to 1M. Default is 50 Ohm
    
    """

    # Calculate Frequency
    # if len(waveform_int)<=16384:
    #     waveform_duration = dt_ns*16384
    # else:
    #     waveform_duration = dt_ns*262144
    waveform_frequency = 1/(waveform_duration)*1e9       
    
    
    if RESET:
        funcgen.write("*RST"); # Reset the function generator
        
        
    # funcgen.write(":DISP OFF");         # Output the selected arb waveform  
    funcgen.write_binary_values(f":DATA{ch}:DAC VOLATILE,", waveform_int, datatype="h", is_big_endian=True)    

    if INIT:
        
        funcgen.write(f":FUNCtion{ch}:USER VOLATILE");         # Select the active arb waveform
        funcgen.write(f":FUNCtion{ch}:SHAPe USER");         # Output the selected arb waveform        
        
        # Interpolation ON
        if interpolation:
            funcgen.write(":DIG:TRAN:INT ON")    
        else:
            funcgen.write(":DIG:TRAN:INT OFF")   


        funcgen.write(f":FREQuency{ch} {waveform_frequency}");  # Set the frequency to 125kHz so that 1 sample = 1ns. 122070 = 2Ghz/16384. 2.5 GHz for 81160



        # Set Trigger mode
        funcgen.write(f":ARM:SOUR{ch} {trigger_mode}");  
        funcgen.write(f":ARM:FREQ{ch} {trigger_freq}Hz")    


        # Set output parameter
        funcgen.write(f":OUTPut{ch}:LOAD output_impedance");               # Output termination is 50 Ohms
        funcgen.write(f":VOLT{ch}:OFFS {output_offset}VPP");  


        funcgen.write(":DISP ON");         
        funcgen.write(f":OUTPut{ch} ON");      # Output the selected arb waveform  
        

    # Set amplitude
    funcgen.write(f":VOLT{ch} {output_voltage}VPP"); 
        
    
def trigger(funcgen):
    funcgen.write(":TRIG")

def mc_exp_decay(n_expected, n_threshold, WLS_t_decay, LASER_time_spread, N_EXPERIMENTS =4000, N_PLOTS=200,
                 SEED=1, PULSE=None, Fs=2.5, noise_voltage_density = 1e-9):

    # # Signal size and trigger
    # n_expected = [n_low, n_high]# number of eh expected
    # n_threshold = 4.5 # threshold [eh]

    # # WLS fiber parameter
    # WLS_t_decay = 2.7 # [ns]
    # # Laser parameter
    # LASER_time_spread = 0.15 # [ns], sigma, Gaus wavelet


    # # Run parameters
    # N_EXPERIMENTS = 10_000
    # N_PLOTS = 200
    # SEED = 1

    PULSE_DURATION = len(PULSE)/Fs
    TIME_SERIES = np.linspace(0,PULSE_DURATION,len(PULSE))
    RNG = np.random.default_rng(seed=SEED)

    results_trigger_time = []
    results_pulses = []
    for i in tqdm(range(N_EXPERIMENTS)):
        # Get the actual numer of photons by Sampling from poisson
        if type(n_expected) is not list:
            RNG.poisson(lam=n_expected)
        else:
            n_gen = RNG.integers(n_expected[0], n_expected[1])
 
        if n_gen>0:
            pulse_this = np.zeros_like(PULSE)
            # Get the time of each photon by sampling from Gauss
            # time_gen = RNG.normal(loc=LASER_time_spread*7, scale=LASER_time_spread*7, size=n_gen)
            time_gen = 0
 
            # Get the decay time of each photon by sampling from exponential
            time_decay = time_gen + RNG.exponential(scale=WLS_t_decay, size=n_gen)
 
            for iphoton in range(n_gen):
                if time_decay[iphoton]>0 and time_decay[iphoton]<PULSE_DURATION:
                    # Roll and pile one more pulse to the final pulse
                    pulse_this+=roll_zeropad(PULSE, int(time_decay[iphoton]*Fs))
 
            results_trigger_time.append(np.argmax(pulse_this>(n_threshold*np.max(PULSE))))
    
            # Add in noise
            noise_trace = hp.band_limited_noise(0, 1.5e9, samples=len(PULSE), samplerate=int(Fs*1e9))* noise_voltage_density
            pulse_this+=noise_trace
 
            if i<N_PLOTS:
                results_pulses.append(pulse_this)
            

             
    results_trigger_time=np.array(results_trigger_time)   
    results_pulses=np.array(results_pulses)
    
    return results_trigger_time, results_pulses, TIME_SERIES




def ssa3000_gettrace(scope):
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
    
    n_queries = 100
    while n_queries>0:
        if scope.query("*OPC?")=='1\n':
            break    
            
    scope.write(":TRACe? 1")
    trace_raw = scope.read_raw()[:-1] # delete the last \n char
    return np.frombuffer(trace_raw, dtype=np.float32)
    
    
    
