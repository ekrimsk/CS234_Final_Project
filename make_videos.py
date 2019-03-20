
# TODO -- remove unneeded imports 
# TODO -- move out to methods -- maybe even add into utils later  
# TODO -- clean up and add comments 

# TODO == environement unfairly terminating right now for greedy case (goal can be too close to overshoot)


import os
import sys
import time
import shlex
from subprocess import check_call
import glob             # for getting files 
import subprocess       # for ffmpeg scripts 
import numpy as np



def main():

    # TODO -- either add in a way to parse arguments for what it should process or make callabler 


    # TODO = make these seperate meothds we call so main is short and readable -- maybe even add to utilities module??? 

    #check_call('mkvmerge -o trained.mp4 ./test_video/openaigym.video.0.trained_00000.video000000.mp4 + ./test_video/openaigym.video.1.trained_00001.video000000.mp4 + ./test_video/openaigym.video.2.trained_00002.video000000.mp4 ' , shell=True, executable='/bin/bash')
    record_path = 'trained_video'
    trained_vids = sorted(glob.glob(record_path + '/*trained*.mp4'), key=os.path.getmtime)
    greedy_vids = sorted(glob.glob(record_path + '/*greedy*.mp4'), key=os.path.getmtime)




    compare_vids = ["{0}/side_by_side{1:05d}.mp4".format(record_path,i) for i in range(len(trained_vids))]
    print(compare_vids)
    
    proc_list = []
    for i in range(len(trained_vids)):
        vid_train = trained_vids[i]
        vid_greedy = greedy_vids[i]
        # need to add thing  to overwrite or delete existing beofew

        command =   'ffmpeg '  + \
                    ' -i ' + vid_train + \
                    ' -i ' + vid_greedy + \
                    ' -filter_complex \'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]\' ' + \
                    ' -map [vid] ' + \
                    ' -c:v libx264 ' + \
                    ' -crf 23 ' + \
                    ' -preset veryfast ' + compare_vids[i]
        #print("com: ", command)               

        # NOTE: might not want/need close_fds=True arg             
        args = shlex.split(command)
        proc_list.append(subprocess.Popen(args, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))

        print('Combining ',  vid_train , " and " , vid_greedy)
    
    for p in proc_list:
        try:
            outs, errs = p.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            p.kill()
            outs, errs = p.communicate()
            print('timeou--------------------t')


    print('Indivual comparison vids make')


    outname = record_path + '/compare.mp4' #TODO -- better naming conve
    mkv_command = 'mkvmerge -o ' + outname + " " 
    for vid in compare_vids:
        mkv_command += vid + " + "
    mkv_command = mkv_command[0:-2]
    print(mkv_command)
    mkvargs = shlex.split(mkv_command)

    #f = subprocess.Popen(mkvargs, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print('finished combined vid')
    check_call(mkv_command, shell=True, executable='/bin/bash')
    

if __name__ == '__main__':
    main()


