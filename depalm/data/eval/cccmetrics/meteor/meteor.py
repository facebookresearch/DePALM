#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import subprocess
import threading

from ..cccpaths import JAVA_PATH, METEOR_JAR


class MeteorScorer:
    def __init__(self):
        self.meteor_cmd = [JAVA_PATH, '-jar', '-Xmx2G', METEOR_JAR,
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd,
            # cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, references, predictions):
        assert(len(references) == len(predictions))
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in range(len(references)):
            assert(type(references[i]) is list)
            assert(type(predictions[i]) is str)
            assert(len(references[i]) >= 1)
            stat = self._stat(predictions[i], references[i])
            eval_line += ' ||| {}'.format(stat)

        self._write(eval_line)

        for i in range(len(references)):
            scores.append(float())
        score = float(self._read())
        self.lock.release()

        return score, scores

    def _write(self, line):
        self.meteor_p.stdin.write(f'{line}\n'.encode())
        self.meteor_p.stdin.flush()
    
    def _read(self):
        return self.meteor_p.stdout.readline().decode().strip()
    
    def _communicate(self, line):
        self._write(line)
        return self._read()

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        return self._communicate(score_line)

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        stats = self._communicate(score_line)
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats 
        score = float(self._communicate(eval_line))
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self._read())
        self.lock.release()
        return score
 
    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
