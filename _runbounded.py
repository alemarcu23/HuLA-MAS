import subprocess, sys, time
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 600
cfg = sys.argv[2] if len(sys.argv) > 2 else '/tmp/config_seeded.yaml'
with open('/tmp/run.log', 'w') as out:
    p = subprocess.Popen([sys.executable, 'main.py', '--config', cfg],
                         stdout=out, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < LIMIT:
        if p.poll() is not None:
            print(f'exited rc={p.returncode} after {time.time()-t0:.0f}s'); break
        time.sleep(2)
    else:
        print(f'reached {LIMIT}s, terminating'); p.terminate()
        try: p.wait(10)
        except subprocess.TimeoutExpired: p.kill()
