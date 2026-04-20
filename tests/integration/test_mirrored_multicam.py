

def test_mirrored_multicam_defaults(run_cli, compare_to_golden, tmpdir, pytestconfig, request):

    output_dir = run_cli(
        subcommand='mirrored-multicam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # unneeded computation
        camera_names=['top', 'bot'],
    )
    compare_to_golden(request.node.name, output_dir)


def test_mirrored_multicam_fixed_smooth_param(
    run_cli, compare_to_golden, tmpdir, pytestconfig, request,
):

    output_dir = run_cli(
        subcommand='mirrored-multicam',
        input_dir=str(pytestconfig.rootpath / 'data' / 'mirror-mouse'),
        output_dir=tmpdir,
        bodypart_list=['paw1LH', 'paw2LF'],  # unneeded computation
        camera_names=['top', 'bot'],
        s=10,
    )
    compare_to_golden(request.node.name, output_dir)
