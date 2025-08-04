import os
import joblib
from turftopic import KeyNMF
from topicwizard.figures import topic_map
import kaleido # for saving

def main():

    # load model
    m = joblib.load(os.path.join("obj", "topic_models", "NEe_50.pkl"))

    # create figure
    fig = topic_map(m)

    # change appearance of circles and text
    trace = fig.data[0]
    trace.marker.color = 'lightblue'
    trace.marker.colorscale = None
    trace.marker.showscale = False
    trace.marker.line = dict(
        color='skyblue',
        width=2
    )
    trace.textposition = 'middle center'
    trace.textfont = dict(size=12)

    # change plot dimensions
    fig.update_layout(
        width=1200,
        height=900,
        autosize = False
    )

    # save as html
    path = os.path.join("obj", "plots", "topic_map.html")
    fig.write_html(path)

if __name__ == "__main__":
    main()