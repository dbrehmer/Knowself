import os
from flask import Flask
from __init__ import create_app

app = create_app()


# @app.route('/')
# def hello():
#     return 'Hello World!'

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

