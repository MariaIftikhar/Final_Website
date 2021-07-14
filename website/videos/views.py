from django.shortcuts import render, redirect, get_object_or_404
from .models import Video, UserProfile,PredictedAnomaly
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.contrib import auth
import os
from moviepy.editor import VideoFileClip
from django.http import HttpResponse
from .import myutils
import cv2 
import torch


def UploadView(request):
    if not request.user.is_authenticated:
        return redirect('../login')
    else:
        if request.GET.get('query'):
            return redirect('/videos/?query=' + request.GET.get('query') )
        if request.FILES.get('video'):
            if request.POST.get('title') and request.FILES.get('video') and request.POST.get('description'):
                video = Video()
                video.title = request.POST.get('title')
                video.description = request.POST.get('description')
                video.video = request.FILES.get('video')
                video.user = request.user
                video.save()
                return redirect('/videos/' + str(video.id))

        return render(request, './upload.html', {})

def VideosView(request):

    queryset_list = Video.objects.all()
    query = request.GET.get('query')
    if query:
        queryset_list = queryset_list.filter(
        Q(title__icontains=query)|
        Q(description__icontains=query)|
        Q(user__username__icontains=query)
        ).distinct()
    paginator = Paginator(queryset_list, 5)
    page = request.GET.get('page')
    queryset = paginator.get_page(page)

   
    return render(request, 'videos.html', {'queryset': queryset})

def Gallery(request):
    queryset = PredictedAnomaly.objects.filter(user_id = request.user.id)
    return render(request,'anomaly.html',{'queryset': queryset})
def GalleryView(request, anomaly_id):
        video = get_object_or_404(PredictedAnomaly, pk = anomaly_id)
        return render(request,'show.html',{'video':video})
def saveFrame(request,video_id): 
 
    video = get_object_or_404(Video, pk = video_id)
    still = PredictedAnomaly.objects.filter(title = video.title)
    print(still)
    if not still:
        path = os.path.dirname(__file__)
        path = path.replace('videos',video.video.url)
        clip = VideoFileClip(path) 
        clip = clip.subclip(0,10)  
        clip.write_videofile("media/videos/"+video.title+".mp4") 
    # clip.ipython_display(width = 360)
        path2vido = "media/videos/"+video.title+".mp4"
        frames, v_len =myutils.get_frames(path2vido, n_frames=29)
        print(len(frames), v_len )
        model_type = "3dcnn"
        model = myutils.get_model(model_type = model_type, num_classes = 2)
        model.eval();
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        path2weights = "weights_3dcnn.pt"
        model.load_state_dict(torch.load(path2weights,map_location={'cuda:0': 'cpu'}))
        model.to(device);
        imgs_tensor = myutils.transform_frames(frames, model_type)
        print(imgs_tensor.shape, torch.min(imgs_tensor), torch.max(imgs_tensor))
        with torch.no_grad():
            out = model(imgs_tensor.to(device)).cpu()
            print(out.shape)
            pred = torch.argmax(out).item()
            print(pred)
        if pred==1:
            anomaly = PredictedAnomaly()
            path1 = "videos/"+video.title+".mp4"
            anomaly.video = path1
            anomaly.title = video.title
            pic = UserProfile.objects.last()
            anomaly.frame1 = pic.picture.url
            anomaly.frame2 = pic.picture.url
            anomaly.user = video.user
            anomaly.save() 
            return redirect('gallery') 
        else:
            html = "<html><body >No Anomaly</body></html>"
            return HttpResponse(html)


    else:
        html = "<html><body >Video Already Predicted</body></html>"

        return HttpResponse(html)

def VideoDetailView(request,video_id):
    video = get_object_or_404(Video, pk = video_id)
    videoUserName = video.user.username
    userprofile = get_object_or_404(UserProfile, username=videoUserName)
    recentvideos =Video.objects.all()
    tempvideos= []
    count = 0
    for recentvideo in recentvideos:
        if recentvideo.id != video_id:
            tempvideos.append(recentvideo)
            count= count +1
            if count >= 4:
                break


    recentvideos = tempvideos

    #Used When Query Search Used in Video Detail View to redirect to the search view.
    if request.GET.get('query'):

        return redirect('/videos/?query=' + request.GET.get('query') )

    videoUserName = video.user.username
    userprofile = get_object_or_404(UserProfile, username=videoUserName)
    #adding a view with every video detail GET request
    video.views = video.views + 1
    video.save()

    return render(request,'videodetail.html',{'video':video,'recentvideos':recentvideos, 'userprofile':userprofile})







