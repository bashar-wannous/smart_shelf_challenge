"""
run server with flask
to run on windows: open terminal in project directory, and type flask run

"""
from flask import Flask, render_template_string
from local_paths import path_to_log
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string("""<!DOCTYPE html>
<html>

<head>
<meta charset="utf-8" />
<title>Test</title>

<script type="text/javascript" src="http://code.jquery.com/jquery-1.8.0.min.js"></script>

<script type="text/javascript">
function updater() {
  $.get('/data', function(data) {
    $('#time').html(data);  // update page with new data
  });
};

setInterval(updater, 100);  // run `updater()` every 1000ms (1s)
</script>

</head>

<body>
Logger: <span id="time"><span>
</body>

</html>""")


@app.route('/data')
def data():
    with open(path_to_log, "r") as f: 
        content = f.read()
    """send current content"""
    return content

if __name__ == "__main__":
    app.run(debug=True)