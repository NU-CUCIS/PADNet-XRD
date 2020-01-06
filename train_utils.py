import subprocess, shlex, time
import numpy as np, re
import time, json, hashlib
import sys, os, datetime
import collections
import smtplib
import os, sys
from os.path import basename
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import subprocess, shlex, time

class Record_Results(object):
    def __init__(self, logfile):
        #print('logfile:', logfile)
        if os.path.isfile(logfile):
            ext = logfile.split('.')[-1]
            filename = logfile[:-len(ext)-1]
            #print('filename: ', filename)
            file_suff = ''
            if '_' in filename:
                file_suff = filename.split('_')[-1]
                filename = filename[:-len(file_suff)-1]
            try:
                if 'v' in file_suff:
                    file_suff = file_suff.remove('v')
                    file_suff = 'v'+str(int(file_suff)+1)
                else:
                    file_suff = file_suff+'_v1'
            except:
                file_suff += get_date_str()
            logfile_c = filename+'_'+file_suff+'.'+ext
            with open(logfile_c, 'w') as f:
                f.write(open(logfile, 'r').read())
        self.logfile = logfile
        #print 'logfile:', logfile
        self.f = open(logfile,'w')
        self.f.close()

    def fprint(self, *stt):
        sto = reduce(lambda x,y: str(x)+' '+str(y), list(stt))
        print sto
        try:
            sto = str(datetime.datetime.now())+':'+ sto
        except: pass
        assert os.path.exists(self.logfile)
        self.f = open(self.logfile, 'a')
        try:
            self.f.write('\n'+sto)
        except: pass
        self.f.close()

    def clear(self):
        self.f = open(self.logfile, 'w')
        self.f.close()
    def close(self):
        print 'no need to close'
        return

def get_date_str():
    datetim = str(datetime.datetime.now()).replace('.','').replace('-','').replace(':','').replace(' ','')[2:14]
    return datetim

def create_dir(direc):
    command = "mkdir -p "+direc
    #print "create dir for ", direc
    if not (os.path.exists(direc) and os.path.isdir(direc)):
        os.system(command)

def run_model(python_model,conffile, logfile, gpu):
    command = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu'+str(int(gpu))+',floatX=float32 nohup python '+python_model+' '+conffile+' > run_'+logfile+' &'
    output, pid = run_command(command)

def run_command(args, output_file = None, gpu_num = -1, timeout = 10):
    args = args.strip()
    if gpu_num >=0:
        print 'setting up gpu:', gpu_num
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    bg_process = False
    print "Command: ", args, '\n'
    stdout_ = subprocess.PIPE
    if '&' in args:
        stdout_ = None
        bg_process= True
        args = 'nohup '+args
    if bg_process:
        os.system(args)
        return '', -1
    args = shlex.split(args)

    p = subprocess.Popen(args, stdout=stdout_, shell=False)#, stdin=None, stdout=PIPE, stderr=None)
    op = ""
    timeout = timeout/10
    if timeout ==0:
        timeout = 1
    pid = p.pid
    if bg_process:
        return '', pid
    time.sleep(1)
    for i in range(timeout):
        if p.poll() is  0:
            op, err = p.communicate(None)
            return op, pid
        else:
            time.sleep(10)
    if p.poll() is not 0:
        p.kill()
        print "Timeout, killed process"
    else:
        op, err = p.communicate(None)
    if output_file:
        open(output_file, 'w').write(op)
    return op, pid

def check_model(hyper_params):
    log_folder = hyper_params['log_folder']
    model_file = hyper_params['models_config']
    try:
        model_file = open(os.path.join(log_folder, model_file), 'r')
        models = model_file.read().strip().split('\n\n')
        md5s = map(lambda x: x.strip().split('\n')[0].strip(), models)
        #print md5s
        if hyper_params['md5_sum'] in md5s:
            return True, None,0
        res_file.close()
    except:
        pass
    return  False, None,0

def execute_job(job, exec_config, need_gpu=False):
    pid = 0
    res = None
    job_config = load_config(job['config_file'])
    num_attempts = 1
    gpu_num = -1
    while res is None and num_attempts:
        num_attempts -=1
        memory_avail = True
        if need_gpu:
            gpu_num = find_free_GPU(exec_config['gpu_num'])
        else:
            memory_avail = memory_available()
            if not memory_avail:
                print 'Memory not available'
                break
        print 'free gpu to be set is: ', gpu_num, ' gpus to use are:', exec_config['gpu_num'], 'need gpu: ', need_gpu
        if (need_gpu and gpu_num in exec_config['gpu_num']) or (not need_gpu and memory_avail):
            log_folder = job_config['log_folder']
            res = analyze_job_logs(job)
            if not res:
                output, pid = run_command(job['job_command'], os.path.join(log_folder,'run_'+job_config['log_file']), gpu_num)
            time.sleep(300)
            res = analyze_job_logs(job)
            print 'initial result is: ', res
        else:
            time.sleep(60)
    if res:
        return res, pid, gpu_num
    else:
        return None, None, gpu_num

def md5sum(data):
    if not type(data)==str:
        return hashlib.md5(json.dumps(data, sort_keys=True)).hexdigest()
    else: hashlib.md5(data).hexdigest()
# Find GPU stats
def find_GPU_stats():
    gpu_output, pid = run_command('nvidia-smi', timeout=5)
    print gpu_output
    outputs = gpu_output.strip().split('Processes')[0]
    outputs =  [x.split('|') for x in gpu_output.strip().split('Processes')[0].split('\n') if 'MiB' in	x]
    values = []
    for out in outputs:
        used,avl =  out[2].replace('MiB','').strip().split('/')
        proc = out[3].strip().split(' ')[0].replace('%','').strip()
        value = {'proc':float(proc), 'avlmem':float(avl), 'usedmem':float(
            used), 'memoryuse':float(used)/float(avl)}
        values.append(value)
    return values

# Find free GPU
def find_free_GPU(gpu_to_use):
    values1 = find_GPU_stats()
    time.sleep(10)
    values2 = find_GPU_stats()
    time.sleep(10)
    values3 = find_GPU_stats()
    minproc = 100
    gpunums1 = []
    for gpunum, val in enumerate(values1):
        if val['proc'] < minproc:
            if values1[gpunum]['proc'] < 50 and values1[gpunum]['memoryuse']\
                    < 0.5:
                gpunums1.append(gpunum)
    print gpunums1
    minproc = 100
    gpunums2 = []
    for gpunum, val in enumerate(values2):
        if val['proc'] < minproc:
            if values2[gpunum]['proc'] < 50 and values2[gpunum]['memoryuse']\
                    < 0.5:
                gpunums2.append(gpunum)
    minproc = 100
    gpunums3 = []
    for gpunum, val in enumerate(values3):
        if val['proc'] < minproc:
            if values3[gpunum]['proc'] < 50 and values3[gpunum]['memoryuse']\
                    < 0.5:
                gpunums3.append(gpunum)
    gpus = set(set(gpunums1).intersection(set(gpunums2))).intersection(set(gpunums3))
    gpus = gpus.intersection(set(gpu_to_use))
    gpus = list(gpus)
    print 'free gpus are: ', gpus
    if len(gpus) > 0:
        gpunum = gpus[0]
        #return  len(values1)- gpunum-1
        return gpunum
    return None

# Find pids of jobs running
def memory_available():
    command_name = 'top -U dkj755 -n 1'
    output, pid = run_command(command_name, timeout=5)
    output =  output.strip().split('KiB Mem :')[1].split('\n')[0].strip()
    mems = [int(x) for x in re.findall(r'\d+',output) if int(x) >100]
    return float(mems[1])/float(mems[0]) >= 0.3

# write to a config file
def write_config(config_filename, config):
    with open(config_filename, 'w') as config_file:
        json.dump(config, config_file)

# load from config file
def load_config(config_filename):
    with open(config_filename) as config_file:
        config = json.load(config_file)
    return  config

def get_job_status(job):
    if not os.path.exists(os.path.join(job['log_folder'], job['log_file'])): return 'start'
    log_data = open(os.path.join(job['log_folder'], job['log_file']), 'r').read()
    if 'done' in log_data: return 'done'
    elif 'start training' in log_data: return 'running'
    else: return 'start'

def create_job_config(job):
    print 'creating job config'
    job_path = job['job_command'].split('python')[1].strip().split()[0]
    job_dir, job_file =  os.path.spilt(job_path)
    job['project'] = job_dir.split('/')[-1].strip()
    if not job['config_file']:
        rand_file = get_date_str()
        job_config = {'log_file':'automated_logs_'+rand_file+'.log'}
        job_config['log_folder'] = job_dir
        config_file = os.path.join(job_config['log_folder'], 'config_'+rand_file+'.config')
        write_config(config_file, job_config)
        job['config_file'] = config_file


def send_email(subject, message, proj_files=[]):
    COMMASPACE = ', '
    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = 'Superbox: ' + subject
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = 'dkj755@superbox'
    msg['To'] = COMMASPACE.join(['dipendra009@gmail.com'])
    msg.preamble = message
    # Assume we know that the image files are all in PNG format
    for fil in proj_files:
        # Open the files in binary mode.  Let the MIMEImage class automatically
        # guess the specific image type.
        fp = open(fil, 'rb')
        part = MIMEApplication(fp.read(), Name=basename(fil))
        fp.close()
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(fil)
        msg.attach(part)

    # Send the email via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.sendmail('dkj755@superbox', ['dipendra009@gmail.com'], msg.as_string())
    s.quit()

def email_status(subject, job_info):
    message = 'Project Results\n\n'
    projects = [job_info[x]['project'] for x in job_info]
    proj_files = []
    for pr in set(projects):
        if os.path.exists(os.path.join('/home/dkj755/AutoML',pr)+ '_results.txt'):
            message += 'Project: ' + pr + '\n'
            message+= open(os.path.join('/home/dkj755/AutoML',pr)+ '_results.txt', 'r').read()+'\n\n'
            open(os.path.join('/home/dkj755/AutoML', pr) + '_results.csv', 'wb').write(open(os.path.join('/home/dkj755/AutoML',pr)+ '_results.txt', 'r').read()+'\n\n')
            proj_files.append(os.path.join('/home/dkj755/AutoML', pr) + '_results.csv')
    send_email(subject, message, proj_files)



#send_email('TEST EMAIL', 'TEST')
