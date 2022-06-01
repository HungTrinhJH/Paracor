import React, { Component } from "react";
import "./ShowLanguage.css";
import RadioButton from "../Form/RadioButton";
class ShowLanguage extends Component {
  handleTypeLanguageChange = (key) => (value) => {
    this.props.languageChangeHandler(value);
  };
  render() {
    const { language } = this.props;
    return (
      <div className="col-2 language mt-3">
        <p className="content__title">Source language</p>

        <div className="language">
          <RadioButton
            label="English"
            onChange={this.handleTypeLanguageChange("language")}
            selected={language === "english"}
            value="english"
            styleClass="mb-3"
          />
          <RadioButton
            label="Vietnamese"
            onChange={this.handleTypeLanguageChange("language")}
            selected={language === "vietnamese"}
            value="vietnamese"
          />
        </div>
      </div>
    );
  }
}

export default ShowLanguage;
