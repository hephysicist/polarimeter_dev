import sys
sys.path.append('../lib')
from depol.DepolarizerMessage_pb2 import *
#sys.path.append("../proto")
#from DepolarizerMessage_pb2 import *
import socket
import struct
import time, datetime
from threading import Thread

from math import modf


def frequency2energy(f,n) :
    # f is the frequency, Hz
    # n is harmonic number
    # R = 2*m_e/(g-2) 
    R = 440.64845866025958 #MeV  // PDG-2021
    F0 = 818924.126 #Hz = 181 801 156 Hz / 222 сепаратрисы
    if f == 0 : return 0.0
    return (f/F0+n)*R


def energy2frequency(E) :
    # E is the energy, MeV
    # R = 2*m_e/(g-2) 
    R = 440.64845866025958 #MeV  // PDG-2021
    F0 = 818924.126 #Hz = 181 801 156 Hz / 222 сепаратрисы
    n = E//R #harmonic number
    print("harmonic number = ", n)
    return F0*(E/R-E//R)

#class depolarizer
class depolarizer:

    #some helper function to send and receive message
    def send(self, message):
        try :
            message.timestamp = int(time.time()*1e9)
            self.sock.send(struct.pack('L',message.ByteSize()))
            self.sock.send(message.SerializeToString())
        except BrokenPipeError:
            self.connect()
            self.sock.send(struct.pack('L',message.ByteSize()))
            self.sock.send(message.SerializeToString())

    def receive(self):
      data = self.sock.recv(8, socket.MSG_WAITALL)
      if not data:
          print("WARNING: Unable to connect")
          self.sock.close()
          self.connect();
          data = self.sock.recv(8)
          if not data:
              print(F"ERROR: Unable to connect to {self.host}:{self.port}")
      s = struct.unpack('L',data)
      #print "message size = ",s[0]
      data = self.sock.recv(s[0], socket.MSG_WAITALL)
      m = DepolarizerMessage()
      try:
        m.ParseFromString(data)
      except:
        print('depolarizer.py : Parse error: ')

      return m

    def send2(self, message):
        try :
            message.timestamp = int(time.time()*1e9)
            self.sock2.send(struct.pack('L',message.ByteSize()))
            self.sock2.send(message.SerializeToString())
        except BrokenPipeError:
            self.connect()
            self.sock2.send(struct.pack('L',message.ByteSize()))
            self.sock2.send(message.SerializeToString())

    def receive2(self):
      data = self.sock2.recv(8, socket.MSG_WAITALL)
      #if not data:
      #    print("WARNING: Unable to connect")
      #    self.sock2.close()
      #    self.connect();
      #    data = self.sock2.recv(8)
      #    if not data:
      #        print(F"ERROR: Unable to connect to {self.host}:{self.port}")
      s = struct.unpack('L',data)
      #print ("message size = ",s[0])
      data = self.sock2.recv(s[0], socket.MSG_WAITALL)
      #print ("message data",len(data))
      m = DepolarizerMessage()
      m.ParseFromString(data)
      return m

    def __init__(self, host, port):
      self.host = host
      self.port = port
      self.connect()
      #self.sock2 = sock2et.sock2et()
      #self.sock2.connect((host, port))
      self.message_id=0
      self.is_fmap = False
      self.fmap = []

    def connect(self):
      self.sock = socket.socket()
      self.sock.connect((self.host, self.port))
      self.sock.settimeout(5)


    #def send(self, message):
    #  send(self.sock, message)

    #def receive(self):
    #  return receive(self.sock)

    def do(self, command):
      m = DepolarizerMessage()
      m.id = self.message_id
      self.message_id = self.message_id + 1
      m.command = command
      self.send(m)
      m = self.receive()
      return m.status == DepolarizerMessage.OK

    def start_scan(self):
      return self.do(DepolarizerMessage.START)

    def stop_scan(self):
      return self.do(DepolarizerMessage.STOP)

    def continue_scan(self):
      return self.do(DepolarizerMessage.CONTINUE)


    def get(self,data_type):
      m = DepolarizerMessage()
      m.id = self.message_id
      self.message_id = self.message_id + 1
      m.command = DepolarizerMessage.GET
      m.data_type = data_type
      self.send(m)
      m = self.receive()
      return m.data

    def set(self,data_type,data):
      m = DepolarizerMessage()
      m.id = self.message_id
      self.message_id = self.message_id + 1
      m.command = DepolarizerMessage.SET
      m.data_type = data_type
      m.data = data;
      self.send(m)
      m = self.receive()
      return m.data

    def get_state(self):
      return int(self.get(DepolarizerMessage.STATE))

    def is_off(self):
        state = self.get_state()
        return not (state & 0x1)

    def is_on(self):
        state = self.get_state()
        return state & 0x1 == 0x1

    def is_scan(self):
        state = self.get_state()
        return state & 0x2 == 0x2

    def get_frequency_by_time(self, time_list ):
        m = DepolarizerMessage()
        m.id = self.message_id
        self.message_id = self.message_id + 1
        m.command = DepolarizerMessage.GET
        m.data_type = DepolarizerMessage.FREQUENCY_BY_TIME
        if type(time_list) is float:
            fp = m.fmap.frequency.add()
            fp.timestamp = int(1e9*time_list) #convert to nanoseconds
        else:
            for  t in time_list:
                fp = m.fmap.frequency.add()
                fp.timestamp = int(1e9*t) 
        self.send(m)
        m = self.receive()
        result = []
        for fp in m.fmap.frequency:
            result.append(fp)
        return result

    def get_frequency(self):
      return self.get(DepolarizerMessage.FREQUENCY)

    def get_initial(self):
      return self.get(DepolarizerMessage.INITIAL)

    def get_final(self):
      return self.get(DepolarizerMessage.FINAL)
      
    def get_step(self):
      return self.get(DepolarizerMessage.STEP)

    def get_speed(self):
      return self.get(DepolarizerMessage.SPEED)

    def get_attenuation(self):
      return self.get(DepolarizerMessage.ATTENUATION)

    def get_harmonic_number(self):
      return self.get(DepolarizerMessage.HARMONIC_NUMBER)

    def get_revolution_frequency(self):
      return self.get(DepolarizerMessage.REVOLUTION_FREQUENCY)

    def get_width(self):
      return self.get(DepolarizerMessage.WIDTH)
    
    def get_fmap(self):
      return self.fmap

    def get_log_level(self):
        return self.get(DepolarizerMessage.LOG_LEVEL)

    def set_depolarizer(self, data):
      return self.set(DepolarizerMessage.DEPOLARIZER, data)

    def set_initial(self,data):
      return self.set(DepolarizerMessage.INITIAL,data)

    def set_final(self,data):
      return self.set(DepolarizerMessage.FINAL,data)
      
    def set_step(self,data):
      return self.set(DepolarizerMessage.STEP,data)

    def set_speed(self,data):
      return self.set(DepolarizerMessage.SPEED,data)

    def set_attenuation(self,data):
      return self.set(DepolarizerMessage.ATTENUATION,data)

    def set_harmonic_number(self,data):
      return self.set(DepolarizerMessage.HARMONIC_NUMBER,data)

    def set_revolution_frequency(self,data):
      return self.set(DepolarizerMessage.REVOLUTION_FREQUENCY,data)

    def set_width(self, data):
      return self.set(DepolarizerMessage.WIDTH,data)

    def set_log_level(self, level):
        return self.set(DepolarizerMessage.LOG_LEVEL, level)

    def log(self):
        self.sock2 = socket.socket()
        self.sock2.connect((self.host, self.port))
        m = DepolarizerMessage()
        mid = 0
        m.id = mid
        m.command = DepolarizerMessage.GET
        m.data_type = DepolarizerMessage.FMAP
        self.send2(m)
        m = self.receive2()
        if m.status == DepolarizerMessage.OK:
            count = 0
            while True:
                m = self.receive2()
                for fm in m.fmap.frequency: 
                    t = datetime.datetime.fromtimestamp(fm.timestamp*1e-9).isoformat(sep=' ', timespec='milliseconds')
                    n = fm.harmonic_number
                    f = fm.frequency
                    E = frequency2energy(f, n)
                    speed = fm.speed
                    speed_kevs = frequency2energy(fm.speed, 0)*1e3
                    if count % 20 == 0:
                        print("#{:^24}   {:^11}    {:^10}     {:^7}      {:6}       {:5}    {:6}   ".format("time", "frequency", "energy", "speed", "speed", "atten", "step"))

                    print("{:24} {:11.3f} Hz {:10.3f} MeV {:7.3f} Hz/s {:6.3f} keV/s {:5} dB {:6.3} Hz".format(t, f, E, speed, speed_kevs, fm.attenuation, fm.step))
                    count += 1
          
    def get_fmap_in_thread(self):
        self.sock2 = socket.socket()
        self.sock2.connect((self.host, self.port))
        m = DepolarizerMessage()
        mid = 0
        m.id = mid
        m.command = DepolarizerMessage.GET
        m.data_type = DepolarizerMessage.FMAP
        self.send2(m)
        m = self.receive2()
        if m.status == DepolarizerMessage.OK:
            while self.is_fmap:
                m = self.receive2()
                for fm in m.fmap.frequency: 
                    #print(fm.timestamp, fm.frequency)
                    self.fmap.append([fm.timestamp, fm.frequency])
        self.is_fmap = False

    def start_fmap(self):
      if not self.is_fmap:
        self.is_fmap = True
        self.fmap_thread = Thread(target=self.get_fmap_in_thread, args=())
        self.fmap_thread.start()
        print("fmap tread started")

    def stop_fmap(self):
      self.is_fmap = False
      self.fmap_thread.join()

    def clear_fmap(self):
      self.fmap = []

    def print_fmap(self):
      for [t,f] in self.fmap:
        print(t, f)

if __name__ == '__main__':
    d = depolarizer('vepp4-spin.inp.nsk.su',9090)
    print("power {}".format("ON" if d.is_on() else "OFF"))
    print("scan {}".format("ON" if d.is_scan() else "OFF"))
    print("attenuation ", d.get_attenuation())
    print("speed {:.3f} Hz".format(d.get_speed()))
    print(" step {:.3f} Hz".format(d.get_step()))
    print("initial frequency {:.3f} Hz".format(d.get_initial()))
    print("harmonic number: {}".format(d.get_harmonic_number()))

def scan(E, speed, attenuation):
    # R = 2*m_e/(g-2) 
    R = 440.64845866025958 #MeV  // PDG-2021
    F0 = 818924.0
    s =  speed/R*F0
    harmonic = E//R
    initial = (E/R-harmonic)*F0
    d.set_harmonic_number(harmonic)
    d.set_attenuation(attenuation)
    d.set_speed(s)
    d.set_initial(initial)
    if d.start_scan():
        print("Started scan from {:.3f} MeV {:>5}, speed {:.2f} keV/s, att. {} dB, harmonic {:d}".format(E,
              "up" if speed>0 else "down", 1000.*speed, int(attenuation), int(harmonic)))

def stop():
    d.stop_scan()

def utug(E, dE, speed, attenuation):
    print("Start utug with central {:.3f} MeV range +-{:.3f} MeV speed {:.2f} keV/s att. {} dB".format(E,dE, speed*1000, attenuation))
    #E - is real energy
    # dE is amplitude of sweeping
    T = 2*dE/speed
    try:
        while True:
            #scan up
            begin_time = time.time()
            scan(E-dE, +abs(speed), attenuation)
            while time.time() < begin_time + T:
                time.sleep(1)
            d.stop_scan()
            #scan down
            begin_time = time.time()
            scan(E+dE, -abs(speed), attenuation)
            while time.time() < begin_time + T:
                time.sleep(1)
            d.stop_scan()
    except KeyboardInterrupt:
        print("Stop utug")
        stop()

