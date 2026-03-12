from openreward.environments import Server

from dapo_math import DAPOMath

if __name__ == "__main__":
    server = Server([DAPOMath])
    server.run()
