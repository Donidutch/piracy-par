import numpy as np

from piracy_par.data.sample import SAMPLE_FILE, load_sample, make_synthetic_weekly


def test_bundled_sample_is_labelled_and_reproducible():
    s = load_sample()
    assert len(s) == 156 and s.index.is_monotonic_increasing
    assert s.dtype.kind == "i" and (s >= 0).all()
    regenerated = make_synthetic_weekly()
    np.testing.assert_array_equal(regenerated["count"].to_numpy(), s.to_numpy())

    from importlib import resources

    first = resources.files("piracy_par.data").joinpath(SAMPLE_FILE).read_text().splitlines()[0]
    assert "SYNTHETIC" in first
