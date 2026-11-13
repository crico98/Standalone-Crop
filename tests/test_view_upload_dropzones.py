import unittest

from view_uploads import VIEW_UPLOAD_FIELDS, unique_view_filename


class ViewUploadDropzoneTests(unittest.TestCase):
    def test_view_fields_map_to_the_existing_angle_index_contract(self):
        self.assertEqual(
            VIEW_UPLOAD_FIELDS,
            (
                ("isometric", "isometric_images", 0),
                ("top", "top_images", 1),
                ("face", "face_images", 2),
            ),
        )

    def test_duplicate_names_are_renamed_without_overwriting(self):
        used_names = set()
        self.assertEqual(
            unique_view_filename("isometric", "same.jpg", used_names),
            "same.jpg",
        )
        self.assertEqual(
            unique_view_filename("isometric", "same.jpg", used_names),
            "isometric_same.jpg",
        )
        self.assertEqual(
            unique_view_filename("top", "same.jpg", used_names),
            "top_same.jpg",
        )
        self.assertEqual(
            unique_view_filename("face", "same.jpg", used_names),
            "face_same.jpg",
        )

    def test_name_collision_matching_is_case_insensitive(self):
        used_names = {"photo.jpg"}
        self.assertEqual(
            unique_view_filename("top", "PHOTO.JPG", used_names),
            "top_PHOTO.JPG",
        )


if __name__ == "__main__":
    unittest.main()
