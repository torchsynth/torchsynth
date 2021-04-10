import sys
from wsgiref.simple_server import make_server
from urllib import parse
import json
import tempfile
import scipy.io.wavfile
import numpy as np
import cgi

import torch
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import static


class WebSynth:
    def __init__(self):
        self.port = 8000
        self.address = "localhost"
        self.setup_statics()

        synthconfig = SynthConfig(batch_size=1, reproducible=False)
        self.synth = Voice(synthconfig)
        self.rendered = []

    def run(self):
        with make_server(self.address, self.port, self) as httpd:
            print(f"Run torchsynth websynth at {self.address, self.port}")
            httpd.serve_forever()

    def __call__(self, environ, start_response):
        """
        This gets called any time there is a new request made to the server
        """

        method = environ["REQUEST_METHOD"]
        if method == "GET":
            return self.get(environ, start_response)

        if method == "POST":
            return self.post(environ, start_response)

        status = "500 Internal Server Error"
        start_response(status, [])
        return []

    def get(self, environ, start_response):
        """
        Process a GET request
        """

        path = environ["PATH_INFO"]
        query = environ["QUERY_STRING"]

        if path in self.statics:

            if path == "/":
                self.synth.randomize()

            response_body = self.statics[path]["data"]()
            start_response(
                "200 OK", self.get_headers(response_body, self.statics[path]["type"])
            )

        elif path == "/random_synth":
            response_body = self.get_random_patch()
            start_response("200 OK", self.get_headers(response_body, "audio/wav"))

        elif path == "/parameters":
            response_body = bytes(self.get_parameters(), "utf-8")
            start_response(
                "200 OK", self.get_headers(response_body, "application/json")
            )

        elif path == "/get_rendered":
            response_body = self.get_rendered()
            start_response("200 OK", self.get_headers(response_body, "audio/wav"))

        else:
            response_body = b""
            start_response(
                "404 Not Found", self.get_headers(response_body, "text/html")
            )

        return [response_body]

    def post(self, environ, start_response):
        """
        Process a POST request
        """

        # the environment variable CONTENT_LENGTH may be empty or missing
        path = environ["PATH_INFO"]
        post_env = environ.copy()
        post_env["QUERY_STRING"] = ""
        post = cgi.FieldStorage(
            fp=environ["wsgi.input"], environ=post_env, keep_blank_values=True
        )

        if path == "/set_patch":
            self.set_patch(post)
            response_body = bytes(json.dumps({"success": True}), "utf-8")
            start_response(
                "200 OK", self.get_headers(response_body, "application/json")
            )

        else:
            response_body = b""
            start_response(
                "404 Not Found", self.get_headers(response_body, "text/html")
            )

        return [response_body]

    def get_random_patch(self):
        """"""
        audio = self.synth()[0].detach().numpy()
        with tempfile.TemporaryFile() as file_handle:
            scipy.io.wavfile.write(
                file_handle, int(self.synth.sample_rate.item()), audio
            )
            wav = file_handle.read()

        return wav

    def get_rendered(self):
        with tempfile.TemporaryFile() as file_handle:
            scipy.io.wavfile.write(
                file_handle, int(self.synth.sample_rate.item()), self.rendered
            )
            wav = file_handle.read()

        return wav

    def get_parameters(self):
        params = {}
        for key, param in self.synth.get_parameters().items():
            params["_".join(key)] = {
                "name": key,
                "val": param.from_0to1()[0].detach().item(),
                "min": param.parameter_range.minimum,
                "max": param.parameter_range.maximum,
            }
        return json.dumps(params)

    def set_patch(self, patch):
        new_params = {}
        for name in patch:
            key = tuple(name.split("-"))
            new_params[key] = torch.tensor([float(patch[name].value)])

        self.synth.set_parameters(new_params)
        self.rendered = self.synth()[0].detach().numpy()

    @staticmethod
    def get_headers(response_body, mime_type):
        """
        Return headers for HTML
        """

        response_headers = [
            ("Content-Type", mime_type),
            ("Content-Length", str(len(response_body))),
        ]

        return response_headers

    def setup_statics(self):
        """
        Creates a dictionary of static files and lambda functions that return
        binary data when that static file is required
        """
        self.statics = {
            "/": {
                "type": "text/html",
                "data": lambda: pkg_resources.read_binary(static, "index.html"),
            },
            "/js/torchsynth-websynth.js": {
                "type": "text/javascript",
                "data": lambda: pkg_resources.read_binary(
                    static, "torchsynth-websynth.js"
                ),
            },
        }


def main():
    websynth = WebSynth()
    websynth.run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
