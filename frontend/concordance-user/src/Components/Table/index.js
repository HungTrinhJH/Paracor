import React, { Component } from "react";
import "./Table.css";
import { connect } from "react-redux";
import { createAction } from "../../Redux/Action";
import { FETCH_DETAIL_SENTENCE } from "../../Redux/Action/type";
import { dataService } from "../../Services";

class Table extends Component {
  handleSentenceDetail = (id) => {
    let lang = this.props.language === "english" ? "en" : "vn";
    dataService
      .fetchData_SentenceDetail(id, lang)
      .then((res) => {
        let data = {};
        if (this.props.language === "vietnamese") {
          data.source = res.data.target;
          data.target = res.data.source;
        } else data = res.data;
        this.props.dispatch(createAction(FETCH_DETAIL_SENTENCE, data));
        this.props.openModalHandler();
      })
      .catch((err) => {
        alert(err.message);
      });
  };

  render() {
    const { data, selectedID } = this.props;
    let index = data.findIndex((item) => item.sentence_id === selectedID);
    let sentenceFull = null;
    if (index !== -1 && selectedID !== null) {
      sentenceFull = (
        <span className="sentence-full">
          : {data[index].left}
          <span className="text-danger font-weight-bold">
            {data[index].key}
          </span>
          {data[index].right}
        </span>
      );
    }

    let tableContent = this.props.data.map((item) => {
      let activeClass =
        this.props.selectedID === item.sentence_id ? "text-info" : null;
      return (
        <tr
          key={item.sentence_id}
          onClick={() => {
            this.props.handleRowSelected(item.sentence_id);
          }}
          className={activeClass}
          onDoubleClick={() => this.handleSentenceDetail(item.sentence_id)}
        >
          <td className="text-center col-1">{item.sentence_id}</td>
          <td className="text-right col-4">
            {item.left.length < 40 ? item.left : item.left.slice(0, 40) + "..."}
          </td>
          <td className="text-center text-primary col-3">
            {item.key.length < 10 ? item.key : item.key.slice(0, 10) + "..."}
          </td>
          <td className="text-left col-4">
            {item.right.length < 40
              ? item.right
              : item.right.slice(0, 40) + "..."}
          </td>
        </tr>
      );
    });

    return (
      <div className="source-languege">
        {/* Title language */}
        <p className="language-title">
          <i className="fa fa-language mr-2" />
          {this.props.languageTitle} {sentenceFull}
        </p>
        {/* Table content */}
        <div className="table-content">
          <table className="table table-fixed">
            <thead className="text-center">
              <tr>
                <th className="col-1">Id</th>
                <th className="col-4">Left</th>
                <th className="col-3">Key</th>
                <th className="col-4">Right</th>
              </tr>
            </thead>
            <tbody className={this.props.classID}>{tableContent}</tbody>
          </table>
        </div>
      </div>
    );
  }
}
const mapStateToProps = (state) => {
  return {
    language: state.Controller.language,
  };
};
export default connect(mapStateToProps)(Table);
