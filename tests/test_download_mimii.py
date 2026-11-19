"""Small synthetic fixtures for archive integrity and resumable downloading."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile


spec = importlib.util.spec_from_file_location(
    "download_mimii", Path(__file__).resolve().parents[1] / "scripts/download_mimii.py"
)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data / "metadata").mkdir()

    def archive(self, filename="pump/id_00/normal/00000000.wav"):
        archive = self.data / "0_dB_pump.zip"
        with ZipFile(archive, "w") as package:
            package.writestr(filename, b"synthetic audio fixture")
        return {
            "key": archive.name, "size": archive.stat().st_size,
            "checksum": "md5:" + hashlib.md5(archive.read_bytes()).hexdigest(),
            "links": {"self": "https://example.invalid/archive"},
        }

    def test_extract_verify_and_preserve_existing_archive(self):
        entry = self.archive()
        downloader.process(entry, self.data)
        self.assertEqual((self.data / "pump_0db/pump/id_00/normal/00000000.wav").read_bytes(),
                         b"synthetic audio fixture")
        self.assertTrue((self.data / entry["key"]).exists())
        downloader.process(entry, self.data)

    def test_reject_bad_checksum_before_extraction(self):
        entry = self.archive()
        entry["checksum"] = "md5:" + "0" * 32
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            downloader.process(entry, self.data)
        self.assertFalse((self.data / "pump_0db").exists())

    def test_reject_archive_traversal(self):
        entry = self.archive("pump/../../outside.wav")
        with self.assertRaisesRegex(ValueError, "Unexpected archive path"):
            downloader.process(entry, self.data)
        self.assertFalse((self.data / "outside.wav").exists())

    def test_preserve_different_existing_recording(self):
        entry = self.archive()
        existing = self.data / "pump_0db/pump/id_00/normal/00000000.wav"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"keep this recording")
        with self.assertRaisesRegex(ValueError, "Existing file differs"):
            downloader.process(entry, self.data)
        self.assertEqual(existing.read_bytes(), b"keep this recording")

    def test_resume_contiguous_prefix(self):
        archive = self.data / "archive.part"
        archive.write_bytes(b"abc")

        def response(request, timeout):
            self.assertEqual(request.headers["Range"], "bytes=3-5")
            result = io.BytesIO(b"def")
            result.status = 206
            result.headers = {"Content-Range": "bytes 3-5/6"}
            return result

        with patch.object(downloader, "urlopen", response):
            downloader.download({"key": "archive", "size": 6,
                                 "links": {"self": "https://example.invalid/archive"}}, archive)
        self.assertEqual(archive.read_bytes(), b"abcdef")


if __name__ == "__main__":
    unittest.main()
