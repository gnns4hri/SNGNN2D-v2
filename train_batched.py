import os, sys, time

try:
    while True:
        os.system(' python3 utils/train_batched_script.py ')
        print('JUST FINISHED A TRAINING TASK. Waiting 10 secs.')
        time.sleep(10)
except KeyboardInterrupt:
    sys.exit(1)
