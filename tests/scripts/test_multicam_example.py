

def test_multicam_example_defaults(run_script, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'multicam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse-separate'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # , 'paw3RF', 'paw4RH'],  # unneeded computation
        camera_names=['top', 'bot'],
    )
    compare_to_golden(request.node.name, output_dir)


def test_multicam_example_fixed_smooth_param(
    run_script, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'multicam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse-separate'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # , 'paw3RF', 'paw4RH'],  # unneeded computation
        camera_names=['top', 'bot'],
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)


def test_multicam_example_defaults_nonlinear(
    run_script, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'multicam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'fly'),
        output_dir=tmpdir,
        bodypart_list=['L1A', 'L1B'],
        camera_names=['Cam-A', 'Cam-B', 'Cam-C'],
        calibration=str(pytestconfig.rootpath / 'data' / 'fly' / 'calibration.toml'),
    )
    compare_to_golden(request.node.name, output_dir)


def test_multicam_example_fixed_smooth_param_nonlinear(
    run_script, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_script(
        script_file=str(pytestconfig.rootpath / 'scripts' / 'multicam_example.py'),
        input_dir=str(pytestconfig.rootpath / 'data' / 'fly'),
        output_dir=tmpdir,
        bodypart_list=['L1A', 'L1B'],
        camera_names=['Cam-A', 'Cam-B', 'Cam-C'],
        calibration=str(pytestconfig.rootpath / 'data' / 'fly' / 'calibration.toml'),
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)
