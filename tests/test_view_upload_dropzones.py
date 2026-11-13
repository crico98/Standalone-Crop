from io import BytesIO

from app import app, _collect_view_uploads


def test_view_dropzones_generate_explicit_manifest_and_unique_names():
    data = {
        "isometric_images": [
            (BytesIO(b"iso-a"), "same.jpg"),
            (BytesIO(b"iso-b"), "same.jpg"),
        ],
        "top_images": [(BytesIO(b"top"), "same.jpg")],
        "face_images": [(BytesIO(b"face"), "same.jpg")],
    }

    with app.test_request_context(
        "/upload",
        method="POST",
        data=data,
        content_type="multipart/form-data",
    ):
        files, manifest = _collect_view_uploads()

    assert [item["angle_index"] for item in manifest] == [0, 0, 1, 2]
    assert [item["file_path"] for item in manifest] == [
        "same.jpg",
        "isometric_same_2.jpg",
        "top_same.jpg",
        "face_same.jpg",
    ]
    assert [file.filename for file in files] == [
        "same.jpg",
        "isometric_same_2.jpg",
        "top_same.jpg",
        "face_same.jpg",
    ]


def test_view_dropzones_return_empty_when_no_files_are_selected():
    with app.test_request_context(
        "/upload",
        method="POST",
        data={},
        content_type="multipart/form-data",
    ):
        files, manifest = _collect_view_uploads()

    assert files == []
    assert manifest == []
