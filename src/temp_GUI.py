import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import queue
from datetime import datetime

class SimpleCANMessage:
    def __init__(self, can_id: int, data: bytes):
        self.timestamp = datetime.now()
        self.can_id = can_id
        self.dlc = len(data)
        self.data = data

    def to_row(self):
        ts = self.timestamp.strftime('%H:%M:%S.%f')[:-3]
        data_hex = ' '.join(f'{b:02X}' for b in self.data)
        return (ts, f'0x{self.can_id:03X}', str(self.dlc), data_hex)

class AnomalyDetector:
    def is_anomaly(self, msg: SimpleCANMessage):
        spoof_ids = {0x700, 0x7FF}
        dos_ids = {0x100}

        if msg.can_id in spoof_ids:
            return True, 'Spoofing Attack (RPM/Gear/Speed gauge manipulation)'
        if msg.can_id in dos_ids:
            return True, 'Denial-of-Service (DoS)'

        counts = {}
        for b in msg.data:
            counts[b] = counts.get(b, 0) + 1
        top_count = max(counts.values())
        if top_count >= len(msg.data) - 1 and len(msg.data) >= 4:
            return True, 'Fuzzy/Flooding Attack'

        s = sum(msg.data)
        if s > 2000 or s < 5:
            return True, 'Spoofing Attack (RPM/Gear/Speed gauge manipulation)'

        return False, ''

class CANSimulator(threading.Thread):
    def __init__(self, out_queue: queue.Queue, detector: AnomalyDetector, rate_hz=5):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.detector = detector
        self.running = False
        self.rate_hz = rate_hz
        self.inject_next_as_anom = False
        self.paused = False
        # attack control
        self.attack_mode = False
        self.attack_type = 'All'  # 'All', 'DoS', 'Fuzz', 'Spoof'

    def run(self):
        self.running = True
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            if self.attack_mode:
                # when in attack mode generate anomaly according to selected type
                msg = self._generate_anomalous_message(forced_type=self.attack_type)
            elif self.inject_next_as_anom:
                msg = self._generate_anomalous_message()
                self.inject_next_as_anom = False
            else:
                msg = self._generate_normal_message()

            is_anom, anom_type = self.detector.is_anomaly(msg)
            self.out_queue.put((msg, is_anom, anom_type))
            time.sleep(1.0 / max(1, self.rate_hz))

    def stop(self):
        self.running = False

    def inject_anomaly_next(self):
        self.inject_next_as_anom = True

    def toggle_pause(self):
        self.paused = not self.paused

    def set_attack(self, enabled: bool, attack_type: str = 'All'):
        self.attack_mode = enabled
        self.attack_type = attack_type

    def _generate_normal_message(self):
        can_id = random.choice([0x50, 0x100, 0x18F, 0x200, 0x300])
        dlc = random.choice([2, 4, 6, 8])
        data = bytes(random.randint(0, 200) for _ in range(dlc))
        return SimpleCANMessage(can_id, data)

    def _generate_anomalous_message(self, forced_type=None):
        # forced_type: None or 'All' or 'DoS'/'Fuzz'/'Spoof'
        if forced_type is None or forced_type == 'All':
            mode = random.choice(['dos', 'fuzz', 'spoof'])
        else:
            m = forced_type.lower()
            if 'dos' in m:
                mode = 'dos'
            elif 'fuzz' in m:
                mode = 'fuzz'
            else:
                mode = 'spoof'

        if mode == 'dos':
            can_id = 0x100
            data = bytes([random.randint(200, 255) for _ in range(8)])
        elif mode == 'fuzz':
            can_id = random.choice([0x200, 0x300, 0x18F])
            # fuzz represented as flooding of 0xFF
            data = bytes([0xFF] * 8)
        else:  # spoof
            can_id = random.choice([0x700, 0x7FF])
            # spoof messages may mimic realistic ranges
            data = bytes(random.randint(0, 200) for _ in range(8))
        return SimpleCANMessage(can_id, data)

class CANGuiApp:
    def __init__(self, root):
        self.root = root
        root.title('CANBus Anomaly Detection — Real-Time')
        root.geometry('1100x640')

        header = tk.Frame(root)
        header.pack(fill='x', padx=8, pady=8)
        tk.Label(header, text='Machine Learning Vigilance: CANBus Anomaly Detection', font=('Helvetica', 16, 'bold')).pack(side='left')
        tk.Label(header, text='Real-Time Monitoring', font=('Helvetica', 10, 'italic')).pack(side='left', padx=10)

        ctrl = tk.Frame(root)
        ctrl.pack(fill='x', padx=8, pady=6)
        self.start_btn = tk.Button(ctrl, text='Start Monitor', command=self.start_sim)
        self.start_btn.pack(side='left')
        self.stop_btn = tk.Button(ctrl, text='Stop Monitor', command=self.stop_sim, state='disabled')
        self.stop_btn.pack(side='left', padx=6)
        self.pause_btn = tk.Button(ctrl, text='Pause', command=self.toggle_pause, state='disabled')
        self.pause_btn.pack(side='left', padx=6)
        self.inject_btn = tk.Button(ctrl, text='Inject Anomaly', command=self.inject_anom, state='disabled')
        self.inject_btn.pack(side='left', padx=6)

        # Attack controls: select type and start/stop attacking
        tk.Label(ctrl, text='Attack Type:').pack(side='left', padx=(20,4))
        self.attack_type_cb = ttk.Combobox(ctrl, values=['All', 'DoS', 'Fuzz', 'Spoof'], state='readonly')
        self.attack_type_cb.set('All')
        self.attack_type_cb.pack(side='left')
        self.attack_btn = tk.Button(ctrl, text='Start Attacking', command=self.toggle_attack, state='disabled')
        self.attack_btn.pack(side='left', padx=6)

        tk.Button(ctrl, text='Clear', command=self.clear_table).pack(side='left', padx=6)

        stats = tk.Frame(root)
        stats.pack(fill='x', padx=8, pady=6)
        self.normal_var = tk.IntVar(value=0)
        self.anom_var = tk.IntVar(value=0)
        tk.Label(stats, text='Normal:').pack(side='left')
        tk.Label(stats, textvariable=self.normal_var, width=6, anchor='w').pack(side='left')
        tk.Label(stats, text='Anomaly:').pack(side='left', padx=(20, 0))
        tk.Label(stats, textvariable=self.anom_var, width=6, anchor='w').pack(side='left')

        table_frame = tk.Frame(root)
        table_frame.pack(fill='both', expand=True, padx=8, pady=6)
        cols = ('Time', 'CAN ID', 'DLC', 'Data (hex)', 'Status', 'Anomaly Type')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=18)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140 if c != 'Data (hex)' else 420)
        vsb = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        self.tree.pack(side='left', fill='both', expand=True)

        footer = tk.Label(root, text='Green = Normal. Red = Anomaly with type shown.', anchor='w')
        footer.pack(fill='x', padx=8, pady=(0, 8))

        self.msg_queue = queue.Queue()
        self.detector = AnomalyDetector()
        self.simulator = CANSimulator(self.msg_queue, self.detector, rate_hz=5)
        self.updater_running = False
        self.attacking = False

    def start_sim(self):
        if not self.simulator.is_alive():
            self.simulator = CANSimulator(self.msg_queue, self.detector, rate_hz=5)
            self.simulator.start()
        else:
            self.simulator.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.pause_btn.config(state='normal')
        self.inject_btn.config(state='normal')
        self.attack_btn.config(state='normal')
        if not self.updater_running:
            self.updater_running = True
            self.root.after(100, self._process_queue)

    def stop_sim(self):
        self.simulator.stop()
        # ensure attack mode disabled
        self.simulator.set_attack(False)
        self.attacking = False
        self.attack_btn.config(text='Start Attacking')

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.pause_btn.config(state='disabled')
        self.inject_btn.config(state='disabled')
        self.attack_btn.config(state='disabled')

    def toggle_pause(self):
        self.simulator.toggle_pause()
        self.pause_btn.config(text='Resume' if self.simulator.paused else 'Pause')

    def inject_anom(self):
        self.simulator.inject_anomaly_next()

    def toggle_attack(self):
        # Toggle attack mode on simulator
        self.attacking = not self.attacking
        chosen = self.attack_type_cb.get() or 'All'
        self.simulator.set_attack(self.attacking, attack_type=chosen)
        self.attack_btn.config(text='Stop Attacking' if self.attacking else 'Start Attacking')

    def clear_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.normal_var.set(0)
        self.anom_var.set(0)

    def _process_queue(self):
        processed = 0
        try:
            while True:
                msg, is_anom, anom_type = self.msg_queue.get_nowait()
                self._add_message_to_table(msg, is_anom, anom_type)
                processed += 1
        except queue.Empty:
            pass
        if self.simulator.running or processed > 0:
            self.root.after(100, self._process_queue)
        else:
            self.updater_running = False

    def _add_message_to_table(self, msg, is_anom, anom_type):
        row = msg.to_row()
        status = 'ANOMALY' if is_anom else 'NORMAL'
        tag = 'anom' if is_anom else 'norm'
        self.tree.insert('', 0, values=(*row, status, anom_type), tags=(tag,))
        self.tree.tag_configure('anom', background='#FFBDBD')
        self.tree.tag_configure('norm', background='#D6FFD6')
        if is_anom:
            self.anom_var.set(self.anom_var.get() + 1)
        else:
            self.normal_var.set(self.normal_var.get() + 1)

if __name__ == '__main__':
    root = tk.Tk()
    app = CANGuiApp(root)
    root.protocol('WM_DELETE_WINDOW', lambda: (app.simulator.stop(), root.destroy()))
    root.mainloop()
