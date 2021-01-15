function getBathValue(){
    var uiBathrooms = document.getElementsByName("uiBathrooms");
    for(var i in uiBathrooms){
        if(uiBathrooms[i].checked){
            return parseInt(i)+1;
        }
    }
    return -1;
}

function getBHKValue(){
    var uiBHK = document.getElementsByName("uiBHK");
    for(var i in uiBHK){
        if(uiBHK[i].checked){
            return  parseInt(i)+1; 
        }
    }
    return -1;

}


function onClickedEstimatedPrice(){
    console.log("Estimated price button clicked");
    var sqft = document.getElementById("uiSqft");
    var bhk = getBHKValue();
    var bathrooms= getBathValue();
    var location = document.getElementById("uiLocations");
    var estprice = document.getElementById("uiEstimatedPrice");

    console.log(location.value,sqft.value, bhk, bathrooms);  

  //  var url="http://127.0.0.1:5000/predict_home_price";
  var url ="/api/predict_home_price";
    $.post(url,{
        total_sqft: parseFloat(sqft.value),
        location: location.value,
        bhk:bhk,
        bath: bathrooms
       
    },function(data,status){
        console.log(data.estimated_price);
        estprice.innerHTML= "<h2>" + data.estimated_price.toString() + "Lakh</h2>";
        console.log(status);

    });


}


function onPageLoad(){
    console.log("document loaded");
   // var url = "http://127.0.0.1:5000/get_location_names";
   var url ="/api/get_location_names";
    $.get(url,function(data,status) {
        console.log("got response for get_location_names request");
        if(data){
            console.log("got data");
            var locations=  data.locations;
            if(locations)
            {
                console.log("got location");
            }
            var uiLocations = document.getElementById("uiLocations");
            console.log(uiLocations);
            $('#uiLocations').empty();
           
            for(var i in locations){
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });  
}

window.onload = onPageLoad;