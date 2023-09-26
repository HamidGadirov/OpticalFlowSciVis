import subprocess
import platform

class Args:
    """arguments for autotuning"""
    elem_t = 'unsigned char'
    winSize = 15
    joint_warp_buf_update = 1
    blockDim = 256
    gridDim = 64
    buf_size = 32
    logFName = 'log.csv'
    def write(self, fname):
        f = open(fname, 'w')
        print('write to ' + fname)
        f.write('// generated automatically by mlzss_tune.py\n')
        f.write('#ifndef __MLZSS_PARAMS__\n')
        f.write('#define __MLZSS_PARAMS__\n')
        f.write('#include <string>\n')
        f.write('#include "mlzss_args.h"\n')
        f.write('#include "mlzss_std_args.h"\n')
        f.write('const unsigned int winSize = '+str(self.winSize)+';\n')
        f.write('const unsigned int buf_size = '+str(self.buf_size)+';\n')
        f.write('const bool joint_warp_buf_update = '+str(self.joint_warp_buf_update)+';\n')
        f.write('const dim3 cuBlockDim('+str(self.blockDim)+');\n')
        f.write('const dim3 cuGridDim('+str(self.gridDim)+');\n')
        f.write('const std::string log_fname("'+self.logFName+'");\n')
        f.write('typedef '+self.elem_t+' elem_t;\n')
        # if self.elem_t == 'unsigned char':
        #     f.write('#define BASE_T UCHAR\n')
        # else:
        #     f.write('#define BASE_T USHORT\n')
        f.write('const unsigned int nSharedMemElems = ' + str(self.buf_size*self.blockDim)+ ';\n')
        f.write('#endif\n')
        


f = open(Args.logFName, 'w')
#write csv header for result data
f.write('winSize, buf_size, joint_up, blockDim, gridDim, time, size\n')
f.close()

args_list = []

# args = Args()
# args_list.append(args)

# args = Args()
# args.buf_size = 64
# args_list.append(args)

# args = Args()
# args.blockDim = 128
# args_list.append(args)

# args = Args()
# args.blockDim = 128
# args.joint_warp_buf_update = 0
# args_list.append(args)

# args = Args()
# args.blockDim = 128
# args.buf_size = 64
# args_list.append(args)

# args = Args()
# args.blockDim = 128
# args.gridDim = 32
# args_list.append(args)

args = Args()
args.gridDim = 32
args_list.append(args)

# args = Args()
# args.gridDim = 32
# args.joint_warp_buf_update = 0
# args_list.append(args)

# args = Args()
# args.blockDim = 64
# args_list.append(args)

# args = Args()
# args.buf_size = 64
# args_list.append(args)

# args = Args()
# args.buf_size = 64
# args.joint_warp_buf_update = 0
# args_list.append(args)

# args = Args()
# args.elem_t = 'unsigned short'
# args.winSize = 15
# args.buf_size = 32
# args_list.append(args)

# args = Args()
# args.elem_t = 'unsigned short'
# args.winSize = 17
# args.buf_size = 64
# args_list.append(args)

# args = Args()
# args.elem_t = 'unsigned short'
# args.winSize = 31
# args.buf_size = 64
# args.blockDim = 128
# args_list.append(args)

# args = Args()
# args.elem_t = 'unsigned short'
# args.winSize = 31
# args.buf_size = 64
# args.blockDim = 256
# args.gridDim = 32
# args_list.append(args)

for a in args_list:
    a.write('mlzss_params_auto.h')
    subprocess.check_call('make')
    input_data_dir = '/home/freysn/tmp/raw/'
    if platform.system() == 'Darwin':
        input_data_dir = '/Users/freysn/tmp/raw/'
    subprocess.check_call(['./mlzss_cuda', input_data_dir])
