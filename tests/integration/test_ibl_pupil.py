

def test_ibl_pupil_defaults(run_cli, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_cli(
        subcommand='ibl-pupil',
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-pupil'),
        output_dir=tmpdir,
    )
    compare_to_golden(request.node.name, output_dir)


def test_ibl_pupil_fixed_smooth_param(
    run_cli, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_cli(
        subcommand='ibl-pupil',
        input_dir=str(pytestconfig.rootpath / 'data' / 'ibl-pupil'),
        output_dir=tmpdir,
        diameter_s=0.99,
        com_s=0.99,
    )
    compare_to_golden(request.node.name, output_dir)
