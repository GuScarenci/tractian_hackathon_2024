import flask
import boto3
import os

# get the database helper functions, non-html routes and session dictionary
import db

s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"], use_ssl=False)

# create the main flask application
app = flask.Flask("Hackaton", "")


@app.route("/")
def root():
    return "working\n"


@app.route("/testread")
def testread():
    id = flask.request.args.get("id")

    if not id:
        ret = []
        for row in db.query("SELECT ID, NAME FROM USERS", None, db.every):
            ret.append({"id": row[0], "name": row[1]})
        return ret

    row = db.query("SELECT ID, NAME FROM USERS WHERE ID = %s", [id], db.one)
    if row is None:
        return flask.Response(status=404)
    else:
        return {"id": row[0], "name": row[1]}


@app.route("/testwrite", methods=["POST"])
def testwrite():
    name = flask.request.form.get("name")
    if name is None:
        return flask.Response(status=400)

    row = db.query("INSERT INTO USERS (NAME) VALUES (%s) RETURNING *", [name], db.one)

    return {"id": row[0], "name": row[1]}


@app.route("/tests3read/<path:key>")
def tests3read(key):
    try:
        obj = s3.get_object(Bucket=os.environ["BUCKET"], Key=key)
        return flask.Response(obj["Body"].read(), mimetype="text/plain")
    except s3.exceptions.NoSuchKey:
        return flask.Response(status=404)


@app.route("/tests3write/<path:key>", methods=["POST"])
def tests3write(key):
    try:
        s3.put_object(Bucket=os.environ["BUCKET"], Key=key, Body=flask.request.get_data())
        return flask.Response(status=201)
    except Exception as e:
        return flask.Response(str(e), status=500)

