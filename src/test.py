import subprocess

config_file = "C:/Users/gcooper/Downloads/opensmile-3.0-win-x64/config/mfcc/MFCC12_0_D_A.conf"

proc = [f"""SMILExtract -C {config_file} -I "out.wav" -O test.csv""",f"""SMILExtract -C {config_file} -I "test.wav" -O test1.csv"""]
procs = []
for p in proc:
    procs.append(subprocess.Popen(p))

for p in procs:
    p.wait()