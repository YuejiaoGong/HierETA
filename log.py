import os
import sys
import time
import zipfile
from tensorboardX import SummaryWriter
project_dir = os.path.dirname(__file__)

def findExt(folder, extensions, exclude_list):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        if any(substring in root for substring in exclude_list):
            continue
        for extension in extensions:
            for filename in filenames:
                if filename.endswith(extension):
                    matches.append(os.path.join(root, filename))
    return matches

def backup_code(outfname, folder, extensions, exclude_list):
    filenames = findExt(folder, extensions, exclude_list)
    zf = zipfile.ZipFile(outfname, mode='w')
    for filename in filenames:
        zf.write(filename)
    zf.close()
    print('saved %i files to %s' % (len(filenames), outfname))

class logger_tb(object):
    def __init__(self, log_dir, description, code_backup, log_tb, time_stamp=True):
        description = description.split(" ")
        if time_stamp == True:
            t_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            description.insert(0, t_stamp)
        m_dir = "_".join(description)
        self.log_dir = os.path.join(project_dir, log_dir, m_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if log_tb:
            self.writer = SummaryWriter(log_dir)
        self.g_step = 0
        if code_backup:
            backup_code(os.path.join(self.log_dir, 'code_snapshot.zip'), '.', ['.py', '.json', '.sh', '.txt', 'json', '.npy'], ['result', '.vscode', 'data-info', '.idea', 'logs'])

    def scalar_summary(self, tag, value):
        self.writer.add_scalar(tag, value, self.g_step)
        self.writer.flush()

    def image_summary(self, tag, image):
        self.writer.add_image(tag, image, self.g_step)
        self.writer.flush()


    def histo_summary(self, tag, values):
        self.writer.add_histogram(tag, values, self.g_step)
        self.writer.flush()

    def graph_summary(self, model, images):
        self.writer.add_graph(model, images)
        self.writer.flush()

class message_logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, "log.txt"), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
