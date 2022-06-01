import React, { useState } from "react";
import { Viewer, Worker } from "@react-pdf-viewer/core";
import { zoomPlugin } from "@react-pdf-viewer/zoom";

// Import styles
import "@react-pdf-viewer/core/lib/styles/index.css";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";
import RadioButton from "../../Components/Form/RadioButton";
// import '@react-pdf-viewer/default-layout/lib/styles/index.css';

// Create new plugin instance

const Help = () => {
  const [lang, setLang] = useState("en");
  const zoomPluginInstance = zoomPlugin();
  const handleChange = (key) => (value) => {
    setLang(value);
  };
  return (
    <div className="myContainer">
      <div className="docs mb-3">
        <h2 className="text-center pt-4 mb-0">Paracor Online Documentation</h2>
      </div>
      <div className=" mb-2 d-flex justify-content-center aligns-item-center">
        <RadioButton
          label="English"
          onChange={handleChange("lang")}
          selected={lang === "en"}
          value="en"
          styleClass="mr-3"
        />
        <RadioButton
          label="Vietnamese"
          onChange={handleChange("lang")}
          selected={lang === "vn"}
          value="vn"
        />
      </div>
      <div>
        <Worker workerUrl="https://unpkg.com/pdfjs-dist@2.6.347/build/pdf.worker.js">
          <div
            style={{
              height: "7  20px",
              width: "100%",
            }}
          >
            <Viewer
              fileUrl={
                lang === "en" ? "/assets/docs_en.pdf" : "/assets/docs_vn.pdf"
              }
              plugins={[
                // defaultLayoutPluginInstance,
                zoomPluginInstance,
              ]}
            />
          </div>
        </Worker>
      </div>
    </div>
  );
};

export default Help;
