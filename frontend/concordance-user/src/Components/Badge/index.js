import React from "react";
import "./Badge.css";
export const Badge = ({ title, deleteHandler }) => {
  return (
    <div className="badge-container">
      <span className="badge-title mr-2">{title}</span>
      <i className="fa fa-times" onClick={()=>deleteHandler(title)}></i>
    </div>
  );
};
