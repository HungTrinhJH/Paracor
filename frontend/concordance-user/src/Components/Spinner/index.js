import React from "react";
import "./spinner.css";
import { Spin } from "antd";
import { connect } from "react-redux";
function Spinner(props) {
  return (
    <div className="spinner-container">
      <Spin size="large" tip={props.msg}></Spin>
    </div>
  );
}

const mapStateToProps = (state) => {
  return {
    msg: state.Controller.loadingMsg,
  };
};
export default connect(mapStateToProps)(Spinner);
