

def test_mirrored_multicam_example_defaults(run_script, tmpdir, pytestconfig):

    run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'mirrored_multicam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # , 'paw3RF', 'paw4RH'],  # unneeded computation
        camera_names=['top', 'bot'],
    )


def test_mirrored_multicam_example_fixed_smooth_param(run_script, tmpdir, pytestconfig):

    run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'mirrored_multicam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # , 'paw3RF', 'paw4RH'],  # unneeded computation
        camera_names=['top', 'bot'],
        s=10
    )
