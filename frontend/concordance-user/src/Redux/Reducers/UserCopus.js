import { ADD_NEW_USER_CORPUS } from "../Action/type";

let initialState = {
  data: [],
};

const UserCorpus = (state = initialState, action) => {
  const { type, payload } = action;
  switch (type) {
    case ADD_NEW_USER_CORPUS:
      state = [...state, ...payload];
      return { ...state };
    default:
      return state;
  }
};

export default UserCorpus;
