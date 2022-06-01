import React from "react";
import { connect } from "react-redux";

const Alert = (props) => {
  return (
    props.msg !== null && (
      <div className="alert alert-danger mt-2">
        <div>
          <i className="fas fa fa-info-circle" />
          <span style={{ marginLeft: "8px" }}>{props.msg}</span>
        </div>
      </div>
    )
  );
};

const mapStateToProps = (state) => {
  return {
    msg: state.Alert.msg,
  };
};
export default connect(mapStateToProps)(Alert);
