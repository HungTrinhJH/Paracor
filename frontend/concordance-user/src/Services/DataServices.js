import axios from "axios";

class DataService {
  baseURL = "http://127.0.0.1:8000/api";
  static adminURL = "http://localhost:3001/login";
  fetchDataHome(number) {
    let enURL = `${this.baseURL}/ensentence/?page=${number}`;
    let vnURL = `${this.baseURL}/vnsentence/?page=${number}`;
    const promise_en = axios.get(enURL);
    const promise_vn = axios.get(vnURL);
    return Promise.all([promise_en, promise_vn]);
  }

  //Fetch Search By Structure
  fetchSearchStructure = (lang, searchList, page) => {
    let url =
      this.baseURL +
      "/search/?structure=" +
      searchList.join("_") +
      `&lang=${lang}`;

    url += `&page=${page}`;
    return axios({
      method: "GET",
      url: url,
    });
  };

  //Fetch Search By Word
  fetchData_Search(searchValue, searchType, lang, { pos, ner }, page) {
    let url = this.baseURL + "/search/?";
    let tagURL = null;
    if (searchType === "phrase") {
      url += `q=${searchValue}&qt=${searchType}&lang=${lang}`;
    } else {
      if (ner) {
        tagURL = `ner=${ner}`;
      }
      if (pos) {
        tagURL = `pos=${pos}`;
      }
      if (pos && ner) {
        tagURL = `pos=${pos}&ner=${ner}`;
      }

      if (searchValue) {
        if (tagURL) {
          url += `q=${searchValue}&qt=${searchType}&lang=${lang}&${tagURL}`;
        } else url += `q=${searchValue}&qt=${searchType}&lang=${lang}`;
      } else url += tagURL;
    }
    url += `&page=${page}`;

    return axios({
      method: "GET",
      url: url,
    });
  }

  // Fetch Search Detail
  fetchData_SentenceDetail = (id, lang) => {
    let urlSentenceDetail = this.baseURL + `/detail/?id=${id}&lang=${lang}`;

    return axios({
      method: "GET",
      url: urlSentenceDetail,
    });
  };

  fetchData_QueryStatistic = (num, lang, count, typeTag, typeTagDetail) => {
    let urlData = this.baseURL + "/statistic/?lang=" + lang;
    if (num !== "all") {
      if (typeTag !== "") {
        urlData += `&size=${count}&${typeTag}=${typeTagDetail}`;
      } else urlData += `&size=${count}`;
    } else if (typeTag !== "") {
      urlData += `&${typeTag}=${typeTagDetail}`;
    }
    return axios({
      method: "GET",
      url: urlData,
    });
  };
  // FETCH Statistic
  fetchStatistic_query = () => {
    return axios({
      method: "GET",
      url: this.baseURL + "/statistic/?lang=en&size=100",
    });
  };

  //Fetch Statistic Query
  fetchStatistic_total = () => {
    return axios({
      method: "GET",
      url: this.baseURL + "/totalstatistics",
    });
  };

  getTotalPage = (dataLength) => {
    dataLength = parseInt(dataLength);
    if (dataLength % 100 === 0) {
      return Math.floor(dataLength / 100);
    }
    return Math.floor(dataLength / 100) + 1;
  };

  // Fetch multi struture
  fetchSearchMultiStructure = (data) => {
    return axios({
      method: "POST",
      url: this.baseURL + "/search/",
      data: data,
    });
  };
}

export default DataService;
