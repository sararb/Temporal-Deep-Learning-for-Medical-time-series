import yaml


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min_ = t % 60
        return '%2d hr %02d min' % (hr, min_)

    elif mode == 'sec':
        t = int(t)
        min_ = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min_, sec)

    else:
        raise NotImplementedError


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg_ : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        config = yaml.load(stream, Loader = yaml.FullLoader)
    return config


