import React from "react";
const LazyPreload = (importStatement) => {
  const Component = React.lazy(importStatement);
  Component.preload = importStatement;
  return Component;
};

export default LazyPreload;
