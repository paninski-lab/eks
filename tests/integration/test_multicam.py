

def test_multicam_defaults(run_cli, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_cli(
        subcommand='multicam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse-separate'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # unneeded computation
        camera_names=['top', 'bot'],
    )
    compare_to_golden(request.node.name, output_dir)


def test_multicam_fixed_smooth_param(run_cli, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_cli(
        subcommand='multicam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse-separate'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # unneeded computation
        camera_names=['top', 'bot'],
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)


def test_multicam_defaults_nonlinear(run_cli, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_cli(
        subcommand='multicam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'fly'),
        output_dir=tmpdir,
        bodypart_list=['L1A', 'L1B'],
        camera_names=['Cam-A', 'Cam-B', 'Cam-C'],
        calibration=str(pytestconfig.rootpath / 'data' / 'fly' / 'calibration.toml'),
    )
    compare_to_golden(request.node.name, output_dir)


def test_multicam_fixed_smooth_param_nonlinear(
    run_cli, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_cli(
        subcommand='multicam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'fly'),
        output_dir=tmpdir,
        bodypart_list=['L1A', 'L1B'],
        camera_names=['Cam-A', 'Cam-B', 'Cam-C'],
        calibration=str(pytestconfig.rootpath / 'data' / 'fly' / 'calibration.toml'),
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)
