import pytest
import sys
from bootstrap.run import run
from bootstrap.lib.options import Options

option_names = [
    'block',  # v_fusion:block, a_fusion:block
    'block_tucker',  # v_fusion:block_tucker, a_fusion:block_tucker
    'tucker',  # v_fusion:tucker, a_fusion:tucker
    'mutan',  # v_fusion:mutan, a_fusion:mutan
    'mlb',  # v_fusion:mlb, a_fusion:mlb
    'mfb',
    'cat_mlp',
    'mlb_tucker',   # v_fusion:mlb, a_fusion:tucker
    'mlb_mutan',   # v_fusion:mlb, a_fusion:mutan
    'mfb_tucker',   # v_fusion:mfb, a_fusion:tucker
    'mfb_mutan',    # v_fusion:mfb, a_fusion:mutan
    'tucker_mlb',
    'tucker_mfb',
    'mutan_mlb',
    'mutan_mfb'
]
def reset_options_instance():
    Options._Options__instance = None
    sys.argv = [sys.argv[0]] # reset command line args

@pytest.mark.parametrize('option_name', option_names)
def test_attention_options(option_name):
    reset_options_instance()
    sys.argv += [
        '-o', f'../options/{option_name}.yaml',
        '--misc.cuda', 'True',
    ]
    try:
        run()
    except:
        print('Unexpected error:', sys.exc_info()[0])
        assert False
    assert True

if __name__ == '__main__':
    test_attention_options('mutan')
    # test_attention_options('san')
    # test_attention_options('ban')



