import time
import random
import sys
import selenium as sl
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import urllib.request
from selenium.webdriver.chrome.options import Options
from NNtraining import*
from face_exctrations import *
#
driver=webdriver.Chrome("chromedriver")


def if_not_matched(driver):

    k=driver.find_elements_by_id("button")
    k[0].click()
    time.sleep(2)
    k=driver.find_element_by_class_name("messenger-composer")                                                                                                                                                                                                                                                                         .find_element_by_class_name("messenger-composer")
    k.send_keys("היי מה נשמע :)?")
    k.send_keys(Keys.ENTER)
    time.sleep(2)

def if_already_matched(driver):
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME,"profile-pill-buttons-button-inner")))
    buttons=driver.find_elements_by_tag_name("button")
    for button in buttons:
        if "MESSAGE" in button.text:
            f_bnt=button
            f_bnt.click()
            break

    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME,"messenger-composer")))
    k=driver.find_elements_by_class_name("messenger-composer")
    k[0].send_keys(("היי מה נשמע :)?"))
    only_liked=len(driver.find_elements_by_class_name("messenger-toolbar-send"))==0
    if only_liked:
        k[0].send_keys(Keys.ENTER)
    else:
        already_matched_box=driver.find_element_by_class_name("messenger-toolbar-send")
        already_matched_box.click()


def setup():
    driver.get("https://www.okcupid.com/doubletake")
    # cookies = pickle.load(open("cookies.pkl", "rb"))
    # for cookie in cookies:
    #     driver.add_cookie(cookie)

    return driver


def Send_generic_msg2all():

    driver.get("https://www.okcupid.com/who-you-like?cf=intros")
    extract_all_likeable_entities()
    send_msg_to_all_likealbe_entities()


def send_msg_to_all_likealbe_entities():
    entities_urls = exctract_urls()
    for i,ent_url in enumerate(entities_urls):
        driver.get(ent_url)
        if_already_matched(driver)
        print("iter ",i)


def extract_all_likeable_entities():
    last_amount_found = exctract_urls()
    next_amout_found = -1
    while last_amount_found != next_amout_found:
        last_amount_found = next_amout_found
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        next_amout_found = exctract_urls()


def remove_cookies_window():

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME,"button")))
    cookies_ac = driver.find_elements_by_tag_name("button")
    button=[cookie for cookie in cookies_ac if cookie.get_attribute("title")=="Accept Cookies"]
    if len(button)!=0:
        button[0].click()

def exctract_urls():
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME,"userrow-bucket-card-link-container")))
    entities= driver.find_elements_by_class_name("userrow-bucket-card-link-container")
    urls=[]
    for entity in entities:
        urls.append(entity.get_attribute("href"))
    return urls


def interactive(driver):
    driver.get("https://www.okcupid.com/doubletake")

    i=1800
    txt="5"
    while(txt!="0"):
        txt=input("Do you like her? 1 for yes , 2 for no.")
        if txt=="1":
            i=save_photos(driver, i,"liked","likes")

        if txt=="2":
            i=save_photos(driver,i,"passed","pass")
np.set_printoptions(suppress=True,precision=3)
def nn_predicts_entity(driver,classifier):
    time.sleep(4)
    elm = driver.find_elements_by_css_selector('[alt="A photo"]')
    current_photos_url = []
    i=0
    for e in elm:
        if "400x400" in e.get_attribute("src"):
            current_photos_url.append(e.get_attribute("src"))
    for img in current_photos_url:
        urllib.request.urlretrieve(img,"current_attemp/predict"+str(i)+".png")
        i += 1

    pred = face_detection_and_nn_forward(classifier)
    print(pred)
    if len(pred)==0:
        pass_or_like(driver, pred,True)
    else:
        pass_or_like(driver, pred)


def face_detection_and_nn_forward(classifier):
    MTCNN_face_detection(1)
    time.sleep(3)
    resizing_images_detect()
    dir_images = os.listdir("current_attemp")
    images_pr = np.zeros((len(dir_images), 150, 150, 3))
    for i, image in enumerate(dir_images):
        images_pr[i, :, :, :] = (cv2.imread("current_attemp/" + image, 1))
        os.remove("current_attemp/"+image)
    pred = classifier.predict(images_pr)
    for i,p in enumerate(pred):
        if p<0.05:
            cv2.imwrite("why_mistake"+str(p)+".png",images_pr[i,:,:,:])
            pred[i]=0.5
    return pred


def pass_or_like(driver, pred,zero_pics=False):
    if zero_pics:
        print("rejeceted")
        driver.find_element_by_class_name("pass" + "-pill-button-inner").click()
        return
    print(pred.mean())
    if  pred.mean() > 0.6:
        print("accepted")
        driver.find_element_by_class_name("likes-pill-button-inner").click()
        return
    else:
        print("rejeceted")
        driver.find_element_by_class_name("pass" + "-pill-button-inner").click()


def save_photos(driver, i,dirc,passOrLike):
    elm = driver.find_elements_by_css_selector('[alt="A photo"]')
    current_photos_url = []
    for e in elm:
        if "400x400" in e.get_attribute("src"):
            current_photos_url.append(e.get_attribute("src"))
    for img in current_photos_url:
        urllib.request.urlretrieve(img, dirc+"/girl_like" + str(i) + ".png")
        i += 1
    k = driver.find_element_by_class_name(passOrLike+"-pill-button-inner")
    k.click()
    return i






def login(email_in,pw_in):
    email=(driver.find_elements_by_class_name("login-fields-field")[0].find_element_by_id("username"))
    pw=(driver.find_elements_by_class_name("login-fields-field")[1].find_element_by_id("password"))
    click=driver.find_element_by_class_name("login-actions").find_element_by_class_name("login-actions-button")
    email.send_keys(email_in)
    pw.send_keys(pw_in)

    click.click()


import bs4


def main():

    setup()
    email=sys.argv[1]
    pw=sys.argv[2]
    # classifier=define_net()
    # classifier.load_weights("weights_curr_m")
    login(email,pw)
    remove_cookies_window()
    driver.get("https://www.okcupid.com/doubletake")

    time.sleep(1)

    for i  in range(5):
        Send_generic_msg2all()
    driver.close()



if __name__=="__main__":
    main()



