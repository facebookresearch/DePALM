from __future__ import division
import os
import subprocess
import json
import numpy as np
import tempfile

from ..cccpaths import JAVA_PATH, SPICE_JAR


TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class SpiceScorer:
    """
    Main Class to compute the SPICE metric 
    """

    def float_convert(self, obj):
        try:
            return float(obj)
        except:
            return np.nan

    def compute_score(self, references, predictions):
        assert(len(references) == len(predictions))
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for image_id in range(len(references)):
            hypo = predictions[image_id]
            ref = references[image_id]

            # Sanity check.
            assert(type(hypo) is str)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : image_id,
              "test" : hypo,
              "refs" : ref
            })

        cwd = os.path.dirname(os.getcwd())
        temp_dir=os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, mode="w")
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir=os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        spice_cmd = [JAVA_PATH, '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
          '-cache', cache_dir,
          '-out', out_file.name,
          '-subset',
          '-silent'
        ]
        subprocess.check_call(spice_cmd, cwd=os.getcwd())

        # Read and process results
        with open(out_file.name) as data_file:    
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
            imgId_to_scores[item['image_id']] = item['scores']
            spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in range(len(references)):
            # Convert none to NaN before saving scores over subcategories
            score_set = {}
            for category,score_tuple in imgId_to_scores[image_id].items():
                score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
            scores.append(score_set)
        return average_score, scores



