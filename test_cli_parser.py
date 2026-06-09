import unittest

from test_api_inference import build_parser


class TestCliParser(unittest.TestCase):
    def test_build_parser_accepts_pdf_and_document_name(self):
        args = build_parser().parse_args(
            [
                "--api",
                "http://127.0.0.1:8001",
                "--pdf",
                "pdf_data/Port Tariff.pdf",
                "--input",
                "input_param.json",
                "--document-name",
                "Port Tariff.pdf",
            ]
        )

        self.assertEqual(args.api, "http://127.0.0.1:8001")
        self.assertEqual(args.pdf, "pdf_data/Port Tariff.pdf")
        self.assertEqual(args.input, "input_param.json")
        self.assertEqual(args.document_name, "Port Tariff.pdf")


if __name__ == "__main__":
    unittest.main()
