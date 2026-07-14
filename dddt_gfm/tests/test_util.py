import pytest
from dddt_gfm.util.io import OSF_download

def test_osf_download():
    assert OSF_download(0, 'downloaded_test.csv', './') == 200