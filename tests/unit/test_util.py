import fsspec
import darknet.py.util as darknet_util

def test_fsspec_split_github_url():
    url, kwargs = darknet_util.fsspec_split_github_url("github://org:repo@sha/path/to/file", None)
    assert url == "github://path/to/file"
    assert kwargs["org"] == "org"
    assert kwargs["repo"] == "repo"
    assert kwargs["sha"] == "sha"


def test_fsspec_cache_open(mocker):
    split_spy = mocker.spy(darknet_util, "fsspec_split_github_url")
    fsspec_open_mock = mocker.patch.object(fsspec, "open", autospec=True)

    of = darknet_util.fsspec_cache_open("github://org:repo@sha/path/to/file")
    assert fsspec_open_mock.return_value == of

    fsspec_open_mock.assert_called_once()
    split_spy.assert_called_once()
