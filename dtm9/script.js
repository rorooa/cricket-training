// script.js (optional for extra functionality)

// Example: Input validation
document.addEventListener("DOMContentLoaded", function () {
    document.querySelector("form").addEventListener("submit", function (e) {
        let height = document.getElementById("height").value;
        let weight = document.getElementById("weight").value;
        let age = document.getElementById("age").value;

        if (!height || !weight || !age || height <= 0 || weight <= 0 || age <= 0) {
            alert("Please enter valid values for height, weight, and age.");
            e.preventDefault();
        }
    });
});
