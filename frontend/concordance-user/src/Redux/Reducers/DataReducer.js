import {
  FETCH_VI_DATA,
  FETCH_EN_DATA,
  FETCH_DETAIL_SENTENCE,
  FETCH_STATIS_SUMARY,
  UPDATE_TOTAL_PAGE_HOME,
  REFRESH_SEARCH_STRUCTURE,
} from "../Action/type";

let initialState = {
  initData: {
    enData: [],
    viData: [],
  },
  detailSentence: null,
  sumaryData: [
    ["en", 645781, 60032, 32468],
    ["vn", 682632, 60032, 18827],
  ],
  searchList: [],
  totalPageHome: 0,
};

const DataReducer = (state = initialState, action) => {
  switch (action.type) {
    case UPDATE_TOTAL_PAGE_HOME:
      state.totalPageHome = action.payload;

      return { ...state };
    case FETCH_EN_DATA:
      state.initData.enData = action.payload;

      return { ...state };
    case FETCH_VI_DATA:
      state.initData.viData = action.payload;
      return { ...state };
    case FETCH_DETAIL_SENTENCE:
      state.detailSentence = action.payload;
      return { ...state };
    case FETCH_STATIS_SUMARY:
      state.sumaryData = action.payload;
      return { ...state };
    case REFRESH_SEARCH_STRUCTURE:
      state.searchStructure = action.payload;
      return { ...state };
    default:
      return state;
  }
};

export default DataReducer;
