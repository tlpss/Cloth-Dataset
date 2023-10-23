import os
import pathlib
import shutil


def create_filtered_symlink_dataset(root_dir_path: str, target_dir_for_symlinks: str, suffixes_to_symlink=[".png"]):
    """creates a "view" on the dataset by creating symlinks to the images with the appropriate suffixes in the dataset.This is useful for labeling: cvat cannot deal with non-image files and we also don't want to label right views manually for example."""
    # traverse all files in the root_dir
    # find all png's
    # create symlinks to the png's in the target_dir_for_symlinks
    # if they have suffix zed.png, ...
    root_dir = pathlib.Path(root_dir_path)
    # check if root_dir exists and has the expected subfolders
    assert root_dir.exists()
    assert (root_dir / "train").exists()
    assert (root_dir / "test").exists()

    # traverse root_dir to find all files
    for root, dirs, files in os.walk(root_dir):
        relative_root_path = os.path.relpath(root, root_dir)
        symlink_dir = os.path.join(target_dir_for_symlinks, relative_root_path)
        print(symlink_dir)
        os.makedirs(symlink_dir, exist_ok=True)
        for relative_file_path in files:
            for suffix in suffixes_to_symlink:
                if relative_file_path.endswith(suffix):
                    # cannot make symlinks, as cvat does not support them.
                    shutil.copy(os.path.join(root, relative_file_path), os.path.join(symlink_dir, relative_file_path))


if __name__ == "__main__":
    """example usage:
    python artf_clothes/scripts/create_cvat_view.py <root_dir_of_dataset> ./cvat_dataset_view -s rgb_zed.png -s rgb_smartphone.png
    """
    import click

    @click.command()
    @click.argument("root_dir_path", type=click.Path(exists=True))
    @click.argument("target_dir_for_symlinks", type=str)
    @click.option(
        "--suffixes_to_symlink",
        "-s",
        multiple=True,
        type=str,
        help="suffixes (including extension) of files to symlink",
    )
    def cli_create_filtered_symlink_dataset(
        root_dir_path: str, target_dir_for_symlinks: str, suffixes_to_symlink=[".png"]
    ):
        create_filtered_symlink_dataset(root_dir_path, target_dir_for_symlinks, suffixes_to_symlink)

    cli_create_filtered_symlink_dataset()
