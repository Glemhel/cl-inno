#!/usr/bin/env python3

import os
import yaml
import subprocess
import shutil
import sys
import time
import signal
import logging

import nvidia_smi

class Daemon:
    def __init__(self, config_file: str = 'daemon_config.yaml'):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_environment()
        self.bandwidth = self.calculate_bandwidth()
        os.environ['BANDWIDTH'] = str(self.bandwidth)

        self.allowed_devices = [int(d) for d in self.config['DEVICES']]
        self.threshold = self.config.get('UTILIZATION_THRESHOLD', 15)
        self.check_interval = self.config.get('CHECK_INTERVAL', 5)
        self.run_duration_hours = self.config.get('RUN_DURATION', 10)
        self.run_duration = self.run_duration_hours * 3600

        self.processes = []
        self.early_termination = False
        self.logger = self.setup_logging()

    def setup_logging(self):
        log_dir = self.config.get('LOG_DIR', './logs')
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)

        log_file = os.path.join(log_dir, 'daemon.log')

        logger = logging.getLogger('DaemonLogger')
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

        return logger

    def load_config(self):
        if not os.path.exists(self.config_file):
            print(f"Configuration file {self.config_file} does not exist.")
            sys.exit(1)
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def setup_environment(self):
        os.environ['MY_IP'] = self.config['MY_IP']
        os.environ['BASE_PORT'] = str(self.config['BASE_PORT'])
        os.environ['EXP_NAME'] = self.config['EXP_NAME']
        os.environ['WANDB_PROJECT_WORKER'] = self.config['WANDB_PROJECT_WORKER']
        os.environ['WANDB_PROJECT_MONITOR'] = self.config['WANDB_PROJECT_MONITOR']
        os.environ['MODEL_NAME'] = self.config['MODEL_NAME']
        os.environ['USE_PRETRAINED_WEIGHTS'] = str(self.config['USE_PRETRAINED_WEIGHTS'])
        os.environ['USE_PEFT_AND_QUANTIZATION'] = str(self.config['USE_PEFT_AND_QUANTIZATION'])

        hf_user_access_token = self.config.get('HF_USER_ACCESS_TOKEN')
        hf_token = self.config.get('HF_TOKEN')
        if hf_user_access_token:
            os.environ['HF_USER_ACCESS_TOKEN'] = hf_user_access_token
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token

        ulimit = self.config.get('ULIMIT', 16384)
        os.system(f"ulimit -n {ulimit}")

    def calculate_bandwidth(self):
        speedtest_json = self.config.get('SPEEDTEST_JSON', 'speedtest.json')
        if os.path.exists(speedtest_json):
            import json
            with open(speedtest_json, 'r') as f:
                speedtest = json.load(f)
                upload_speed = speedtest.get('upload', 0)
                download_speed = speedtest.get('download', 0)
                bandwidth = int(max(1, min(upload_speed, download_speed) / 1e6))
        else:
            bandwidth = 10
        return bandwidth

    def get_gpu_utilizations(self):
        try:
            nvidia_smi.nvmlInit()
            device_count = nvidia_smi.nvmlDeviceGetCount()
            utilizations = []
            for i in range(device_count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                util_rates = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util_rates.gpu
                utilizations.append((i, gpu_util))
            nvidia_smi.nvmlShutdown()
            return utilizations
        except nvidia_smi.NVMLError as err:
            self.logger.error(f"Error querying NVIDIA SMI: {err}")
            nvidia_smi.nvmlShutdown()
            return []

    def wait_for_low_gpu_utilization(self):
        self.logger.info(f"Waiting for GPU utilization of devices {self.allowed_devices} to drop below {self.threshold}%...")
        while True:
            utilizations = self.get_gpu_utilizations()
            if not utilizations:
                time.sleep(self.check_interval)
                continue

            for gpu_index, utilization in utilizations:
                if self.allowed_devices is not None and gpu_index not in self.allowed_devices:
                    continue 
                self.logger.info(f"GPU {gpu_index} Utilization: {utilization}%")
                if utilization < self.threshold:
                    self.logger.info(f"GPU {gpu_index} utilization is below {self.threshold}%.")
                    return gpu_index
            time.sleep(self.check_interval)

    def start_peers(self):
        base_port = self.config['BASE_PORT']
        my_ip = self.config['MY_IP']
        n_peers = self.config['N_PEERS']

        common_arguments = [
            '--model_name', self.config['MODEL_NAME'],
            '--run_id', self.config['EXP_NAME'],
            '--use_pretrained_weights', str(self.config['USE_PRETRAINED_WEIGHTS']),
            '--use_peft_and_quantization', str(self.config['USE_PEFT_AND_QUANTIZATION']),
            '--bandwidth', str(self.bandwidth),
            '--target_batch_size', str(self.config['BATCH_SIZE']),
            '--run_name', self.config['EXP_NAME'],
            '--learning_rate', str(self.config['LEARNING_RATE']),
            '--optimizer_str', self.config['OPTIMIZER']
        ] + self.config.get('COMMON_ARGUMENTS', [])

        initial_peer_id = f"/ip4/{my_ip}/tcp/{base_port}/p2p/QmXmwSiyodXUJwqgjpibkdH3cW8ok5e9cTAamuiq7dnK6o"
        os.environ['INITIAL_PEER_ID'] = initial_peer_id

        run_base_trainer_path = self.config.get('RUN_BASE_TRAINER_PATH', 'run_base_trainer.py')
        run_aux_peer_path = self.config.get('RUN_AUX_PEER_PATH', 'run_aux_peer.py')

        log_dir = self.config.get('LOG_DIR', './logs')
    
        output_dir_prefix = self.config.get('OUTPUT_DIR_PREFIX', 'outputs')

        for i in range(n_peers):
            self.logger.info(f"Setting up peer {i}...")
            peer_id_path = f"peer{i}.id"
            peer_port = base_port + i
            listen_on = f"/ip4/0.0.0.0/tcp/{peer_port}"
            announce_on = f"/ip4/{my_ip}/tcp/{peer_port}"
            peer_output_path = f"{output_dir_prefix}{i}"

            if os.path.exists(peer_output_path):
                shutil.rmtree(peer_output_path)

            os.environ['LISTEN_ON'] = listen_on
            os.environ['ANNOUNCE_ON'] = announce_on

            arguments = common_arguments + [
                '--identity_path', peer_id_path,
                '--host_maddrs', listen_on,
                '--announce_maddrs', announce_on,
                '--output_dir', peer_output_path
            ]
            if i != 0:
                arguments += ['--initial_peers', initial_peer_id]

            peer_log_file = os.path.join(log_dir, f"peer{i}.log")
            os.environ['WANDB_PROJECT'] = self.config['WANDB_PROJECT_WORKER']
            base_trainer_arguments = [run_base_trainer_path] + arguments
            self.logger.info(f"Starting base trainer peer {i} with command:\npython {' '.join(base_trainer_arguments)}")
            with open(peer_log_file, 'w') as f:
                process = subprocess.Popen(['python'] + base_trainer_arguments, stdout=f, stderr=subprocess.STDOUT)
                self.processes.append(process)


    def kill_processes(self):
        self.logger.info("Killing all subprocesses...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Process {process.pid} did not terminate in time. Killing it forcefully.")
                process.kill()
        self.processes.clear()

    def signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}. Initiating early termination.")
        self.early_termination = True

    def run(self):
        signal.signal(signal.SIGUSR1, self.signal_handler)

        while True:
            self.early_termination = False

            suitable_gpu_index = self.wait_for_low_gpu_utilization()

            # Устанавливаем подходящий GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(suitable_gpu_index)
            self.logger.info(f"Set CUDA_VISIBLE_DEVICES to {suitable_gpu_index}")

            self.start_peers()

            self.logger.info(f"Peers are running. Will monitor for {self.run_duration} seconds or until early termination signal is received.")
            current_pid = os.getpid()
            self.logger.info(f"PID of daemon: {current_pid}")
            self.logger.info(f"To stop training without killing daemon use: kill -USR1 {current_pid}")
            self.logger.info(f"To kill daemon use: kill -SIGINT {current_pid}")

            start_time = time.time()
            elapsed_time = 0

            try:
                while elapsed_time < self.run_duration and not self.early_termination:
                    time.sleep(5)
                    elapsed_time = time.time() - start_time
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt. Exiting.")
                self.kill_processes()
                break

            self.kill_processes()

        self.logger.info("Exiting daemon.")

class StreamToLogger(object):
    def __init__(self, logger: logging.Logger, log_level: int = logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

if __name__ == "__main__":
    daemon = Daemon()
    daemon.run()
