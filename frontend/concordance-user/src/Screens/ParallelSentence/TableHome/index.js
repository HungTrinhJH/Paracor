import React, { Component } from "react";
import { dataService } from "../../../Services";
import { createAction } from "../../../Redux/Action";
import { FETCH_DETAIL_SENTENCE } from "../../../Redux/Action/type";
import { connect } from "react-redux";
class TableHome extends Component {
  // Get detail sentence data => to show word alignment
  handleSentenceDetail = (id) => {
    let lang = this.props.language === "english" ? "en" : "vn";

    //Fetch data alignment
    dataService
      .fetchData_SentenceDetail(id, lang)
      .then((res) => {
        this.props.dispatch(createAction(FETCH_DETAIL_SENTENCE, res.data));
        this.props.openModalHandler();
      })
      .catch((err) => {
        alert(err.message);
      });
  };
  
  render() {
    let { data } = this.props;
    let tbodyContent = data.map((item, index) => {
      let activeClass =
        this.props.selectedID === item.sentence_id ? "text-info" : null;
      return (
        <tr
          key={index}
          onClick={() => {
            this.props.handleRowSelected(item.sentence_id);
          }}
          onDoubleClick={() => this.handleSentenceDetail(item.sentence_id)}
          className={activeClass}
        >
          <td className="col-2">{item.sentence_id}</td>
          <td className="col-10">
            {item.sentence.length > 100
              ? item.sentence.slice(0, 100) + "..."
              : item.sentence}
          </td>
        </tr>
      );
    });
    return (
      <div className="table__home">
        <div className="table-content">
          <table className="table table-fixedcd">
            <thead>
              <tr>
                <th className="col-2">ID</th>
                <th className="col-10">Sentence</th>
              </tr>
            </thead>
            <tbody className={this.props.classID}>{tbodyContent}</tbody>
          </table>
        </div>
      </div>
    );
  }
}

export default connect()(TableHome);
