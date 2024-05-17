from pathlib import Path

# Paths (redefine them locally if need to be)
JAVA_PATH = 'java'
_DATA_ROOT = Path(__file__).parents[4].resolve() / 'depalm-data'
CORENLPPATH = str(_DATA_ROOT)
STANFORD_CORENLP_3_4_1_JAR = str(_DATA_ROOT / 'aac-metrics/stanford_nlp/stanford-corenlp-3.4.1.jar')
SPICE_JAR = str(_DATA_ROOT / 'aac-metrics/spice/spice-1.0.jar')
METEOR_JAR = str(_DATA_ROOT / 'aac-metrics/meteor/meteor-1.5.jar')

# For local dev on macOS with arm: install openjdk@8 using the x86_64 architecture for brew (https://gist.github.com/progrium/b286cd8c82ce0825b2eb3b0b3a0720a0)
if Path('/usr/local/opt/openjdk@8/bin/java').exists():
    JAVA_PATH = '/usr/local/opt/openjdk@8/bin/java'