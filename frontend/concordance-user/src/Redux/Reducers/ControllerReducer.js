import { LANGUAGE_TYPE, RESET_LOADING, UPDATE_PAGE_HOME } from "../Action/type";

let initState = {
  language: "english",
  currentPageHome: 1,
  loaded: false,
  loadingMsg: null,
};

const ControllerReducer = (state = initState, action) => {
  switch (action.type) {
    case LANGUAGE_TYPE:
      state.language = action.payload;
      return { ...state };
    case UPDATE_PAGE_HOME:
      state.currentPageHome = action.payload;
      return { ...state };
    case RESET_LOADING:
      let { loaded, msg } = action.payload;
      state.loaded = loaded;
      state.loadingMsg = msg;
      return { ...state };
    default:
      return state;
  }
};

export default ControllerReducer;
