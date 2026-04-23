document.querySelector(".search-button-field").addEventListener("focusin", () => {
   const event = new CustomEvent("readthedocs-search-show");
   document.dispatchEvent(event);
});
