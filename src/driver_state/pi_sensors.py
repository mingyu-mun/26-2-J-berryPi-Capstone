"""Physical sensor adapters. Imports of serial/SMBus are deferred for PC tests."""
from __future__ import annotations

import glob
import json
import math
import time

from .streaming_io import Sample


class SensorError(RuntimeError):
    pass


class SampleGap(SensorError):
    pass


class SerialDecoder:
    """Explicit ADC/CSV/JSON formats; never guess a baud rate or column meaning."""

    def __init__(self, config):
        self.config = config
        self.last_device = None
        self.last_seq = None
        self.anchor = None
        self.last_receive = None
        self.index = 0

    def parse(self, line: str, received: float) -> Sample:
        cfg = self.config
        kind = cfg['format']
        if kind == 'json':
            obj = json.loads(line)
            value = float(obj[cfg.get('value_field', 'adc')])
            device = obj.get(cfg.get('time_field', 't_us'))
            seq = obj.get(cfg.get('sequence_field', 'seq'))
            lead_off = obj.get(cfg.get('lead_off_field', 'lead_off'), False)
        elif kind in ('adc', 'csv'):
            parts = line.strip().split(',')
            if kind=='adc' and len(parts)!=1:
                raise ValueError('adc format requires exactly one ADC value per line')
            fields=cfg.get('expected_fields')
            if kind=='csv' and fields is not None and len(parts)!=fields:
                raise ValueError('serial CSV field count differs from configured firmware format')
            cols = cfg.get('columns', {'value': 0})
            value = float(parts[cols['value']])
            device = float(parts[cols['time']]) if 'time' in cols else None
            seq = int(parts[cols['sequence']]) if 'sequence' in cols else None
            lead_off = parts[cols['lead_off']].strip() if 'lead_off' in cols else False
        else:
            raise ValueError('serial.format must be adc, csv, or json')
        if value == -1:
            lead_off = True
        if not math.isfinite(value) or (value != -1 and not 0 <= value <= cfg['adc_max']):
            raise ValueError('ADC value out of range or nonfinite')
        if isinstance(lead_off, str):
            if lead_off.lower() not in ('0', '1', 'false', 'true'):
                raise ValueError('invalid lead-off flag')
            lead_off = lead_off.lower() in ('1', 'true')
        elif lead_off not in (False, True, 0, 1):
            raise ValueError('invalid lead-off flag')
        if self.last_receive is not None and received - self.last_receive > cfg['gap_s']:
            raise SampleGap('ECG receive gap')
        if seq is not None:
            seq = int(seq)
            if self.last_seq is not None and seq != self.last_seq + 1:
                raise SampleGap('ECG sequence discontinuity or device restart')
        if cfg['clock'] == 'device':
            if device is None:
                raise ValueError('device timestamp required')
            device = float(device) * cfg.get('time_scale', 1e-6)
            if not math.isfinite(device) or device<0:
                raise ValueError('invalid device timestamp')
            if self.last_device is not None:
                if device <= self.last_device or device - self.last_device > cfg['gap_s']:
                    raise SampleGap('ECG device clock reset/wrap/gap')
            if self.anchor is None:
                self.anchor = received - device
            stamp = self.anchor + device
            if stamp > received + .05:
                raise SampleGap('ECG clock drift: sample would be in the future')
            self.last_device = device
        elif cfg['clock'] == 'sample_count':
            if self.anchor is None:
                self.anchor = received
            stamp = self.anchor + self.index / cfg['fs_hz']
            if stamp > received + cfg['gap_s']:
                raise SampleGap('ECG configured sample rate does not match acquisition')
            if received - stamp > 1.:
                raise SampleGap('ECG sample clock is falling behind; check fs_hz or serial backlog')
        else:
            raise ValueError('serial.clock must be device or sample_count')
        self.last_seq, self.last_receive = seq, received
        self.index += 1
        return Sample(stamp, value, not bool(lead_off))


def serial_reader(config, emit, state, stop, clock=time.monotonic, serial_factory=None):
    if serial_factory is None:
        import serial
        serial_factory = serial.Serial
    while not stop.is_set():
        stream = None
        try:
            port = config['port']
            if port == 'auto':
                candidates = sorted(glob.glob('/dev/serial/by-id/*'))
                if len(candidates) != 1:
                    raise SensorError('set serial.port: auto requires exactly one /dev/serial/by-id device')
                port = candidates[0]
            stream = serial_factory(port, config['baudrate'], timeout=.1)
            if stop.wait(config.get('reset_wait_s',2.)):
                break
            if hasattr(stream,'reset_input_buffer'):
                stream.reset_input_buffer()
            decoder = SerialDecoder(config)
            opened, last_good, bad = clock(), None, 0
            while not stop.is_set():
                raw = stream.readline(256)
                now = clock()
                if raw:
                    try:
                        if not raw.endswith(b'\n'):
                            raise ValueError('incomplete or oversized serial line')
                        sample = decoder.parse(raw.decode('ascii').strip(), now)
                    except SampleGap:
                        raise
                    except (ValueError, KeyError, IndexError, UnicodeError, TypeError):
                        bad += 1
                        if bad >= 20:
                            raise SensorError('20 malformed ECG lines; check format/baudrate')
                        continue
                    bad, last_good = 0, now
                    state('ecg', True, 'valid serial sample')
                    emit('ecg', sample)
                timeout=2. if last_good is not None else 5.
                if now - (last_good if last_good is not None else opened) > timeout:
                    raise SensorError('ECG sample timeout')
        except Exception as exc:
            state('ecg', False, str(exc))
        finally:
            if stream is not None:
                stream.close()
        stop.wait(1.)


class MAX30102:
    """100 Hz red+IR FIFO polling; overflow is an explicit data loss error."""
    def __init__(self, bus, config):
        self.bus, self.config = bus, config
        self.address = config['address']
        self.anchor = None
        self.count = 0

    def read(self, register):
        return self.bus.read_byte_data(self.address, register)

    def write(self, register, value):
        self.bus.write_byte_data(self.address, register, value)

    def configure(self):
        if self.read(0xFF) != 0x15:
            raise SensorError('MAX30102-compatible part ID 0x15 not found')
        self.write(0x09, 0x40)
        for _ in range(100):
            if not self.read(0x09) & 0x40:
                break
            time.sleep(.002)
        else:
            raise SensorError('MAX30102 reset timeout')
        for register in (0x02, 0x03, 0x04, 0x05, 0x06):
            self.write(register, 0)
        self.write(0x08, 0)  # one sample per measurement, no FIFO rollover
        self.write(0x0A, 0x27)  # Match uploaded pipeline: 4096 nA, 100 Hz, 411 us / 18 bit
        self.write(0x0C, self.config['led_current'])
        self.write(0x0D, self.config['led_current'])
        self.write(0x09, 0x03)  # SpO2 red + IR (does not calculate SpO2)

    def poll(self, now):
        wr, rd, overflow = self.read(0x04) & 31, self.read(0x06) & 31, self.read(0x05) & 31
        if overflow:
            raise SampleGap('MAX30102 FIFO overflow')
        count = (wr-rd) & 31
        if not count:
            return []
        data=[]
        remaining=count
        while remaining:
            take=min(remaining,5)  # SMBus block limit is 32 bytes
            part=self.bus.read_i2c_block_data(self.address,0x07,take*6)
            if len(part)!=take*6:
                raise SampleGap('short MAX30102 FIFO read')
            data.extend(part)
            remaining-=take
        if self.anchor is None:
            self.anchor=now-(count-1)/100.
        else:
            # The sensor has no exposed sample timestamp. Follow slow oscillator
            # drift using FIFO arrival time, bounded below one sample per poll.
            phase=now-(self.anchor+(self.count+count-1)/100.)
            if abs(phase)>.3:
                raise SampleGap('MAX30102 polling/clock discontinuity')
            self.anchor+=max(-.001,min(.001,.05*phase))
        result=[]
        for i in range(count):
            v=data[i*6:(i+1)*6]
            red=((v[0]<<16)|(v[1]<<8)|v[2])&0x3FFFF
            ir=((v[3]<<16)|(v[4]<<8)|v[5])&0x3FFFF
            stamp=self.anchor+self.count/100.
            if stamp>now+.05 or now-stamp>.3:
                raise SampleGap('MAX30102 sampling clock drift or stalled polling')
            # Match the uploaded acquisition code: contact is IR threshold.
            # A saturated sample is still a received sample, not a USB/I2C loss;
            # waveform/model quality is evaluated separately downstream.
            good=self.config['finger_threshold']<=ir
            result.append(Sample(stamp,float(ir),good))
            self.count+=1
        return result


def ppg_reader(config, emit, state, stop, clock=time.monotonic, bus_factory=None):
    if bus_factory is None:
        from smbus2 import SMBus
        bus_factory=SMBus
    while not stop.is_set():
        bus=None
        try:
            bus=bus_factory(config['bus'])
            sensor=MAX30102(bus,config)
            sensor.configure()
            last=clock()
            while not stop.is_set():
                now=clock()
                values=sensor.poll(now)
                if values:
                    state('ppg',True,'valid I2C FIFO data')
                    for sample in values:
                        emit('ppg',sample)
                    last=now
                if now-last>1.:
                    raise SensorError('MAX30102 sample timeout')
                stop.wait(.01)
        except Exception as exc:
            state('ppg',False,str(exc))
        finally:
            if bus is not None:
                bus.close()
        stop.wait(1.)
