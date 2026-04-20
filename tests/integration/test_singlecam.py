

def test_singlecam_defaults(run_cli, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_cli(
        subcommand='singlecam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-pupil'),
        output_dir=tmpdir,
    )
    compare_to_golden(request.node.name, output_dir)


def test_singlecam_fixed_smooth_param(run_cli, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_cli(
        subcommand='singlecam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-pupil'),
        output_dir=tmpdir,
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)
